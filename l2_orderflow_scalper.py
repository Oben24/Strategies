from strategy_base import StrategyBase
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta

try:
	import MetaTrader5 as mt5
except Exception:
	mt5 = None


class L2OrderflowScalper(StrategyBase):
	"""L2 Scalping with Consolidation + Order-Flow Confirms (MT5)

	Notes:
	- Uses multi-TF consolidation (approximated on primary feed) + L2/DOM confirms when available
	- DOM/L2 inputs are expected optionally via tick data dict (data[symbol]['dom'] with bids/asks levels)
	- Two structured plays: Breakout-with-Absorption and Retest-and-Go
	- Exit/management signals should be actioned by the engine layer
	"""

	def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
		super().__init__(name, symbols, params)

		# Box params
		self.atr_len = params.get('atr_len', 14)
		self.box_5m_bars = params.get('box_5m_bars', 7)
		self.box_1m_bars = params.get('box_1m_bars', 10)
		self.box_1m_cap_atr = params.get('box_1m_cap_atr', 0.5)
		self.box_5m_cap_atr = params.get('box_5m_cap_atr', 0.6)
		self.bb_len = params.get('bb_len', 20)
		self.bb_dev = params.get('bb_dev', 2.0)
		self.bb_bandwidth_cap_mult = params.get('bb_bandwidth_cap_mult', 0.60) # <= 0.60 × median
		self.adx_len = params.get('adx_len', 14)
		self.adx_flat_cap_5m = params.get('adx_flat_cap_5m', 20)

		# Vol state
		self.atr_below_sma_bars = params.get('atr_below_sma_bars', 3)

		# L2 thresholds
		self.i1_thresh = params.get('i1_thresh', 0.15)
		self.i5_thresh = params.get('i5_thresh', 0.35)
		self.microprice_skew_ticks = params.get('microprice_skew_ticks', 0.15)
		self.max_spread_ticks = params.get('max_spread_ticks', 2)
		self.exit_spread_ticks = params.get('exit_spread_ticks', 3)
		self.tick_size = params.get('tick_size', 0.0001)

		# Volume confirmations
		self.break_vol_mult = params.get('break_vol_mult', 1.8)
		self.retest_vol_mult = params.get('retest_vol_mult', 1.2)
		self.vol_avg_len = params.get('vol_avg_len', 20)

		# Stops
		self.sl_atr_min = params.get('sl_atr_min', 1.0)
		self.sl_atr_max = params.get('sl_atr_max', 1.5)

		# Time stop (seconds)
		self.time_stop_sec_min = params.get('time_stop_sec_min', 8)
		self.time_stop_sec_max = params.get('time_stop_sec_max', 20)

		# Internal state
		self._recent_break: Dict[str, Optional[Dict[str, Any]]] = {symbol: None for symbol in symbols}
		self.max_history_bars = max(240, self.bb_len + 50)

	def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		for symbol in self.symbols:
			if symbol not in data or self._history[symbol].empty:
				continue
			df = self._history[symbol]
			if len(df) < self.max_history_bars:
				continue

			tick = data[symbol]
			price = (tick['bid'] + tick['ask']) / 2
			spread = tick['ask'] - tick['bid']

			# Session filter (approx)
			if not self._in_session_window(symbol, df.index[-1]):
				continue
			# Auto-pause on sustained wide spread (approx): 5 last minutes
			if self._avg_spread(df, 5) > 2 * self.tick_size:
				continue

			atr1 = self._calculate_atr(symbol, self.atr_len)
			if atr1 is None:
				continue

			# Pre-trade consolidation multi-TF (approximated on single series):
			box = self._multi_tf_box(symbol, df, atr1)
			if not box:
				continue

			# Volatility state: ATR below SMA20 for >=3 bars; and on signal, ATR uptick (checked later)
			if not self._atr_squeeze_state(symbol):
				continue

			# L2/DOM filters (proxies allowed)
			l2 = tick.get('dom') if isinstance(tick, dict) else None
			l2_ok, l2_metrics = self._l2_filters(spread, l2)
			if not l2_ok:
				continue

			# Volume/effort confirmation
			if not self._vol_ready(symbol):
				continue

			# Entry modes
			sig = self._breakout_with_absorption(symbol, df, box, atr1, l2_metrics)
			if sig is None:
				sig = self._retest_and_go(symbol, df, box, atr1, l2_metrics)
			if sig is None:
				continue

			# Build order with structure-based SL
			entry = df.iloc[-1]['close']
			long = sig['long']
			sl = self._initial_sl(entry, box, atr1, long)
			if sl is None:
				continue
			# Spread hard gate before send
			if spread >= self.exit_spread_ticks * self.tick_size:
				continue

			order = {
				'symbol': symbol,
				'type': 'BUY' if long else 'SELL',
				'sl': sl,
				'tp': entry + (self.tick_size if long else -self.tick_size), # TP1 = +1 tick; engine can manage scaling
				'entry_price': entry
			}
			logging.info(f"L2Orderflow {symbol}: {order['type']} entry={entry:.5f} SL={sl:.5f} box=({box['low']:.5f}-{box['high']:.5f}) I1={l2_metrics.get('i1')} I5={l2_metrics.get('i5')}")
			return order
		return None

	# ====== Components ======
	def _avg_spread(self, df: pd.DataFrame, minutes: int) -> float:
		# Requires ask/bid series in history; if not available, approximate with high-low fraction
		try:
			# Placeholder: use last N bars median of (high-low)/2
			seg = df.tail(minutes)
			return float((seg['high'] - seg['low']).median() / 2)
		except Exception:
			return float('inf')

	def _multi_tf_box(self, symbol: str, df: pd.DataFrame, atr1: float) -> Optional[Dict[str, float]]:
		# 5m: last 7 bars range <= 0.6×ATR1m; 1m: last 10 bars range <= 0.5×ATR1m
		if len(df) < max(self.box_5m_bars, self.box_1m_bars) + 5:
			return None
		box5 = df.tail(self.box_5m_bars)
		r5 = box5['high'].max() - box5['low'].min()
		if r5 > self.box_5m_cap_atr * atr1:
			return None
		box1 = df.tail(self.box_1m_bars)
		r1 = box1['high'].max() - box1['low'].min()
		if r1 > self.box_1m_cap_atr * atr1:
			return None
		# 1m BB squeeze bandwidth cap ≤ 0.60 × median(20)
		bb_up = df['close'].rolling(self.bb_len).mean() + self.bb_dev * df['close'].rolling(self.bb_len).std()
		bb_lo = df['close'].rolling(self.bb_len).mean() - self.bb_dev * df['close'].rolling(self.bb_len).std()
		bw = (bb_up - bb_lo).tail(self.bb_len)
		med = bw.median()
		if pd.isna(med) or bw.iloc[-1] > self.bb_bandwidth_cap_mult * med:
			return None
		# ADX flat regime approx: reuse 1m ADX < 20 for last 5
		adx = []
		for i in range(5):
			val = self._calculate_adx(symbol, self.adx_len)
			adx.append(val['adx'] if val else 999)
		if max(adx) >= self.adx_flat_cap_5m:
			return None
		return {'low': float(box1['low'].min()), 'high': float(box1['high'].max())}

	def _atr_squeeze_state(self, symbol: str) -> bool:
		df = self._history[symbol]
		atr_series = []
		for i in range(self.atr_below_sma_bars + 1):
			val = self._calculate_atr_on_df(df.iloc[:-(self.atr_below_sma_bars - i)], self.atr_len)
			if val is None:
				return False
			atr_series.append(val)
		if len(atr_series) < self.atr_below_sma_bars + 1:
			return False
		atr_sma20 = pd.Series(atr_series).rolling(20).mean().iloc[-1]
		if pd.isna(atr_sma20):
			return False
		# last N bars below SMA20
		if not all(a < atr_sma20 for a in atr_series[-self.atr_below_sma_bars:]):
			return False
		return True

	def _l2_filters(self, spread: float, dom: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
		metrics = {}
		# Spread 1–2 ticks
		if spread > self.max_spread_ticks * self.tick_size:
			return False, metrics
		if not dom or 'bids' not in dom or 'asks' not in dom:
			# No DOM -> treat L2 confirms as soft; allow entry but mark metrics empty
			return True, metrics
		bids: List[Tuple[float, float]] = dom['bids'][:5]
		asks: List[Tuple[float, float]] = dom['asks'][:5]
		if not bids or not asks:
			return False, metrics
		B0p, B0s = bids[0]
		A0p, A0s = asks[0]
		# I1
		i1 = (B0s - A0s) / max(B0s + A0s, 1e-9)
		# I5
		sumB = sum(s for _, s in bids)
		sumA = sum(s for _, s in asks)
		i5 = (sumB - sumA) / max(sumB + sumA, 1e-9)
		# Microprice skew vs mid
		mid = (B0p + A0p) / 2
		microprice = (A0p * B0s + B0p * A0s) / max(B0s + A0s, 1e-9)
		skew = microprice - mid
		metrics.update({'i1': i1, 'i5': i5, 'skew': skew})
		# Thresholds direction-agnostic; actual check occurs in triggers per side
		return True, metrics

	def _vol_ready(self, symbol: str) -> bool:
		vol_med = self._get_median_volume(symbol, self.vol_avg_len)
		avg = self._calculate_average_volume(symbol, self.vol_avg_len)
		if vol_med is None or avg is None or avg == 0:
			return False
		# Allow entry only if breakout bar can demonstrate higher effort; verified in trigger
		return True

	def _breakout_with_absorption(self, symbol: str, df: pd.DataFrame, box: Dict[str, Any], atr1: float, l2: Dict[str, Any]) -> Optional[Dict[str, bool]]:
		last = df.iloc[-1]
		vol_avg = self._calculate_average_volume(symbol, self.vol_avg_len) or 0
		if last['close'] > box['high']:
			# Volume multiple
			if last['tick_volume'] < self.break_vol_mult * max(vol_avg, 1e-9):
				return None
			# Absorption flip: require I1/I5 in direction and skew >= threshold
			if l2 and (l2.get('i1', 0) < self.i1_thresh or l2.get('i5', 0) < self.i5_thresh or l2.get('skew', 0) < self.microprice_skew_ticks * self.tick_size):
				return None
			return {'long': True}
		elif last['close'] < box['low']:
			if last['tick_volume'] < self.break_vol_mult * max(vol_avg, 1e-9):
				return None
			if l2 and (l2.get('i1', 0) > -self.i1_thresh or l2.get('i5', 0) > -self.i5_thresh or l2.get('skew', 0) > -self.microprice_skew_ticks * self.tick_size):
				return None
			return {'long': False}
		return None

	def _retest_and_go(self, symbol: str, df: pd.DataFrame, box: Dict[str, Any], atr1: float, l2: Dict[str, Any]) -> Optional[Dict[str, bool]]:
		# After breakout, look for single retest with refusal and vol ≥1.2× avg
		vol_avg = self._calculate_average_volume(symbol, self.vol_avg_len) or 0
		last = df.iloc[-1]
		# Long side
		if last['close'] > box['high']:
			recent = df.iloc[-5:]
			if (recent['low'] <= box['high']).any() and last['tick_volume'] >= self.retest_vol_mult * max(vol_avg, 1e-9):
				# L2 refusal: positive I1/I5 and positive skew
				if l2 and (l2.get('i1', 0) >= self.i1_thresh and l2.get('i5', 0) >= self.i5_thresh and l2.get('skew', 0) >= 0):
					return {'long': True}
		# Short side
		elif last['close'] < box['low']:
			recent = df.iloc[-5:]
			if (recent['high'] >= box['low']).any() and last['tick_volume'] >= self.retest_vol_mult * max(vol_avg, 1e-9):
				if l2 and (l2.get('i1', 0) <= -self.i1_thresh and l2.get('i5', 0) <= -self.i5_thresh and l2.get('skew', 0) <= 0):
					return {'long': False}
		return None

	def _initial_sl(self, entry: float, box: Dict[str, Any], atr1: float, long: bool) -> Optional[float]:
		# Hard stop just inside opposite box edge (1–1.5×ATR typical)
		if long:
			edge = box['low'] + 0.1 * atr1
			atr_stop = entry - self.sl_atr_min * atr1
			return min(edge, atr_stop)
		else:
			edge = box['high'] - 0.1 * atr1
			atr_stop = entry + self.sl_atr_min * atr1
			return max(edge, atr_stop)

	def _in_session_window(self, symbol: str, ts: pd.Timestamp) -> bool:
		# Basic windows; refine per broker timezone
		s = symbol.upper()
		t = ts.time()
		if s in ('EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD'):
			# Overlap approx: 08:00–11:00 and 13:30–16:30
			return dtime(8,0) <= t <= dtime(11,0) or dtime(13,30) <= t <= dtime(16,30)
		if s in ('US100', 'NAS100', 'US500', 'SPX500'):
			return dtime(9,45) <= t <= dtime(11,30) or dtime(14,30) <= t <= dtime(16,0)
		if s in ('XAUUSD', 'XAGUSD'):
			return dtime(8,0) <= t <= dtime(12,0) or dtime(13,30) <= t <= dtime(16,0)
		if s.startswith('WTI') or s.startswith('BRENT'):
			return dtime(13,0) <= t <= dtime(16,0)
		return True
