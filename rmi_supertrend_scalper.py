from strategy_base import StrategyBase
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np

try:
	import MetaTrader5 as mt5
except Exception:
	mt5 = None


class RmiSupertrendScalper(StrategyBase):
	"""RMI + SuperTrend Scalper

	Implements:
	- Session/news/spread gates
	- Volatility floors + expansion
	- Consolidation micro-box with touches and EMA200 side discipline
	- Breakout body + volume spike
	- Directional filters: SuperTrend color and RMI trigger with slope
	- Entries: market or stop-buffer at box edge
	- Initial stop: tighter of SuperTrend line or last swing, but >= 0.6×ATR
	- TP1/TP2 guidance; trailing by SuperTrend after +1R is externalized to engine/profit lock
	"""

	def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
		super().__init__(name, symbols, params)

		# Core lengths
		self.timeframe = params.get('timeframe', (mt5.TIMEFRAME_M3 if mt5 else 0))
		self.confirmation_timeframe = params.get('confirmation_timeframe', (mt5.TIMEFRAME_M5 if mt5 else 0))
		self.atr_len = params.get('atr_len', 14)
		self.adx_len = params.get('adx_len', 14)
		self.ema_len = params.get('ema_len', 200)
		self.rmi_len = params.get('rmi_len', 9)
		self.rmi_mom = params.get('rmi_mom', 4)
		self.rmi_low = params.get('rmi_low', 35)
		self.rmi_high = params.get('rmi_high', 65)
		self.adx_min = params.get('adx_min', 18)

		# SuperTrend params (ATR×mult)
		self.st_atr = params.get('st_atr', 10)
		self.st_mult = params.get('st_mult', 2.5)
		self.use_htf_st_15m = params.get('use_htf_st_15m', False)

		# Consolidation box
		self.box_min_bars = params.get('box_min_bars', 5)
		self.box_max_bars = params.get('box_max_bars', 12)
		self.box_height_atr = params.get('box_height_atr', 0.6)
		self.min_touches_per_side = params.get('min_touches_per_side', 2)

		# Volume spike
		self.vol_median_len = params.get('vol_median_len', 20)
		self.break_vol_mult = params.get('break_vol_mult', 1.5)
		self.break_vol_mult_crypto = params.get('break_vol_mult_crypto', 1.7)

		# Spread gate
		self.spread_cap_atr_pct = params.get('spread_cap_atr_pct', 0.15)

		# Entry buffers and management
		self.body_beyond_box_atr = params.get('body_beyond_box_atr', 0.5)
		self.stop_floor_atr = params.get('stop_floor_atr', 0.6)
		self.stop_buffer_atr = params.get('stop_buffer_atr', 0.10)
		self.stop_swing_lookback = params.get('stop_swing_lookback', 8)

		self.max_history_bars = max(300, self.ema_len + 50)

	def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		for symbol in self.symbols:
			if symbol not in data or self._history[symbol].empty:
				continue
			df = self._history[symbol]
			if len(df) < max(self.max_history_bars, self.ema_len + 25):
				continue

			tick = data[symbol]
			price = (tick['bid'] + tick['ask']) / 2
			spread = tick['ask'] - tick['bid']

			# Session & spread gates (news gate assumed provided externally in engine)
			if not self._in_session_window(df.index[-1]):
				continue
			atr = self._calculate_atr(symbol, self.atr_len)
			if atr is None or atr <= 0:
				continue
			if spread > self.spread_cap_atr_pct * atr:
				continue

			# Volatility floors by market (approx heuristic on symbol name)
			if not self._vol_floor_ok(symbol, atr, price):
				continue
			# ATR expansion
			atr_prev = self._get_indicator_value_n_bars_ago(symbol, 'atr', self.atr_len, 1)
			if atr_prev is None or atr <= atr_prev:
				continue

			# Consolidation box
			box = self._detect_box(df, atr)
			if not box:
				continue

			# HTF bias: EMA200 side within box
			ema200 = self._calculate_sma_on_df(df, self.ema_len)
			if ema200 is None:
				continue

			# SuperTrend color on execution TF
			st = self._supertrend(df, self.st_atr, self.st_mult)
			if st is None:
				continue
			st_color = st['color']

			# Optional momentum gate
			adx = self._calculate_adx(symbol, self.adx_len)
			if adx and adx['adx'] < self.adx_min:
				continue

			# RMI trigger
			rmi, rmi_slope = self._rmi(df, self.rmi_len, self.rmi_mom)
			if rmi is None:
				continue

			last = df.iloc[-1]
			body = abs(last['close'] - last['open'])

			# Long side
			if last['close'] > box['high'] and (last['close'] - box['high']) >= self.body_beyond_box_atr * atr:
				if not (price > ema200 and st_color == 'green'):
					continue
				if not self._rmi_long_ok(rmi, rmi_slope):
					continue
				if not self._vol_spike_ok(symbol):
					continue
				entry = last['close']
				sl = self._initial_stop(df, entry, atr, st_line=st['line'], long=True)
				if sl is None:
					continue
				return {
					'symbol': symbol,
					'type': 'BUY',
					'sl': sl,
					'tp': entry + atr, # placeholder; engine may manage R-multiple exits
				}

			# Short side
			if last['close'] < box['low'] and (box['low'] - last['close']) >= self.body_beyond_box_atr * atr:
				if not (price < ema200 and st_color == 'red'):
					continue
				if not self._rmi_short_ok(rmi, rmi_slope):
					continue
				if not self._vol_spike_ok(symbol):
					continue
				entry = last['close']
				sl = self._initial_stop(df, entry, atr, st_line=st['line'], long=False)
				if sl is None:
					continue
				return {
					'symbol': symbol,
					'type': 'SELL',
					'sl': sl,
					'tp': entry - atr,
				}

		return None

	# ===== Components =====
	def _in_session_window(self, ts: pd.Timestamp) -> bool:
		t = ts.time()
		# Approx London–NY overlap + NY open window: 08:00–11:00 & 13:30–16:00
		from datetime import time as dtime
		return (dtime(8, 0) <= t <= dtime(11, 0)) or (dtime(13, 30) <= t <= dtime(16, 0))

	def _vol_floor_ok(self, symbol: str, atr: float, price: float) -> bool:
		s = symbol.upper()
		if s in ('EURUSD','GBPUSD','USDJPY','AUDUSD','NZDUSD','USDCAD'):
			return atr >= 0.0005
		if s in ('US100','NAS100','US500','SPX500'):
			return atr >= 0.0015 * price
		if s == 'XAUUSD':
			return atr >= 0.0012 * price
		if s in ('BTCUSD','ETHUSD'):
			return atr >= 0.0025 * price
		return True

	def _detect_box(self, df: pd.DataFrame, atr: float) -> Optional[Dict[str, float]]:
		# Scan last K window (prefer tighter if possible)
		for k in range(self.box_min_bars, self.box_max_bars + 1):
			window = df.iloc[-k:]
			high = float(window['high'].max())
			low = float(window['low'].min())
			height = high - low
			if height > self.box_height_atr * atr:
				continue
			# Touches: at least 2 per side
			up_touches = ((window['high'] >= high * 0.9995) | (window['close'] >= high * 0.9995)).sum()
			down_touches = ((window['low'] <= low * 1.0005) | (window['close'] <= low * 1.0005)).sum()
			if up_touches < self.min_touches_per_side or down_touches < self.min_touches_per_side:
				continue
			return {'high': high, 'low': low}
		return None

	def _rmi(self, df: pd.DataFrame, length: int, momentum: int) -> (Optional[float], Optional[float]):
		# RMI implementation: RSI applied to price change over 'momentum' bars
		price = df['close']
		diff = price.diff(momentum)
		gain = diff.where(diff > 0, 0.0).ewm(span=length, adjust=False).mean()
		loss = (-diff.where(diff < 0, 0.0)).ewm(span=length, adjust=False).mean()
		rs = gain / loss.replace(0, np.nan)
		rmi = 100 - (100 / (1 + rs))
		val = float(rmi.iloc[-1]) if not np.isnan(rmi.iloc[-1]) else None
		if val is None:
			return None, None
		slope = val - float(rmi.iloc[-2]) if len(rmi) >= 2 else 0.0
		return val, slope

	def _rmi_long_ok(self, rmi: float, slope: float) -> bool:
		return (rmi > self.rmi_low or rmi > 45) and slope > 0

	def _rmi_short_ok(self, rmi: float, slope: float) -> bool:
		return (rmi < self.rmi_high or rmi < 55) and slope < 0

	def _supertrend(self, df: pd.DataFrame, atr_len: int, mult: float) -> Optional[Dict[str, Any]]:
		# Basic SuperTrend implementation on close/hl2
		high = df['high']
		low = df['low']
		close = df['close']
		# ATR
		hl = high - low
		hc = (high - close.shift()).abs()
		lc = (low - close.shift()).abs()
		tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
		atr = tr.ewm(span=atr_len, adjust=False).mean()
		ml = (high + low) / 2
		upper = ml + mult * atr
		lower = ml - mult * atr
		st = pd.Series(index=close.index, dtype=float)
		color = pd.Series(index=close.index, dtype=object)
		trend_up = np.nan
		trend_dn = np.nan
		for i in range(len(close)):
			if i == 0:
				st.iloc[i] = upper.iloc[i]
				color.iloc[i] = 'red'
				trend_up = lower.iloc[i]
				trend_dn = upper.iloc[i]
				continue
			prev = st.iloc[i-1]
			if close.iloc[i] > trend_dn:
				st.iloc[i] = max(lower.iloc[i], trend_up)
				trend_up = st.iloc[i]
				trend_dn = upper.iloc[i]
				color.iloc[i] = 'green'
			else:
				st.iloc[i] = min(upper.iloc[i], trend_dn)
				trend_dn = st.iloc[i]
				trend_up = lower.iloc[i]
				color.iloc[i] = 'red'
		return {'line': float(st.iloc[-1]), 'color': str(color.iloc[-1])}

	def _initial_stop(self, df: pd.DataFrame, entry: float, atr: float, st_line: float, long: bool) -> Optional[float]:
		# Tighter of SuperTrend or recent swing, but not tighter than 0.6×ATR
		if long:
			pullback = float(df['low'].iloc[-self.stop_swing_lookback:].min())
			isl = max(pullback, st_line)
			floor = entry - self.stop_floor_atr * atr
			return min(isl, floor) if isl > floor else floor
		else:
			pullback = float(df['high'].iloc[-self.stop_swing_lookback:].max())
			isl = min(pullback, st_line)
			floor = entry + self.stop_floor_atr * atr
			return max(isl, floor) if isl < floor else floor

	def _vol_spike_ok(self, symbol: str) -> bool:
		med = self._get_median_volume(symbol, self.vol_median_len)
		if med is None or med == 0:
			return False
		last_vol = self._history[symbol]['tick_volume'].iloc[-1]
		mult = self.break_vol_mult_crypto if symbol.upper() in ('BTCUSD','ETHUSD') else self.break_vol_mult
		return last_vol >= mult * med
