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


class PivotPointScalper(StrategyBase):
	"""Pivot Point Scalping — STRICT PROFILE

	Notes:
	- Uses prior-day pivots (P, R1, R2, S1, S2)
	- Requires strict consolidation around an active pivot level
	- Directional bias: 200-EMA + session VWAP regime
	- Primary trigger + ≥2 confirmations
	- Risk: farther of (pivot buffer 0.40×ATR14) vs (box edge 0.25×ATR14)
	- T1/T2 policy delegated partly to engine/profit manager; we set initial TP=T1
	"""

	def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
		super().__init__(name, symbols, params)

		# Regime/session
		self.timeframe = params.get('timeframe', (mt5.TIMEFRAME_M5 if mt5 else 0))
		self.session_london_open = params.get('session_london_open', '08:00')
		self.session_london_hours = params.get('session_london_hours', 1)
		self.session_ny_open = params.get('session_ny_open', '13:30')
		self.session_ny_hours = params.get('session_ny_hours', 2)
		self.session_ny_last_hour = params.get('session_ny_last_hour', False)

		# Filters
		self.atr_len = params.get('atr_len', 14)
		self.bb_len = params.get('bb_len', 20)
		self.bb_dev = params.get('bb_dev', 2.0)
		self.kc_mult = params.get('kc_mult', 1.5)
		self.ema200_len = params.get('ema200_len', 200)
		self.rsi_len = params.get('rsi_len', 14)
		self.vwap_session = True

		# ATR% bands
		self.atr_pct_min_fx = params.get('atr_pct_min_fx', 0.05)
		self.atr_pct_max_fx = params.get('atr_pct_max_fx', 0.45)
		self.atr_pct_min_crypto = params.get('atr_pct_min_crypto', 0.10)
		self.atr_pct_max_crypto = params.get('atr_pct_max_crypto', 0.90)

		# Spread gate vs planned SL
		self.spread_vs_sl_cap = params.get('spread_vs_sl_cap', 0.10) # spread <= 10% of SL

		# Consolidation box
		self.box_min_bars = params.get('box_min_bars', 12)
		self.box_max_bars = params.get('box_max_bars', 30)
		self.box_height_cap_fx = params.get('box_height_cap_fx', 0.25) # ×ATR
		self.box_height_cap_gi = params.get('box_height_cap_gi', 0.30) # gold/indices
		self.pivot_mid_distance_atr = params.get('pivot_mid_distance_atr', 0.20)
		self.bb_in_kc_min_bars = params.get('bb_in_kc_min_bars', 8)
		self.bb_width_pct_atr_cap = params.get('bb_width_pct_atr_cap', 0.25) # ×ATR
		self.bb_width_pct_price_cap = params.get('bb_width_pct_price_cap', 0.0025) # 0.25%
		self.bb_width_pct_price_cap_crypto = params.get('bb_width_pct_price_cap_crypto', 0.0060) # 0.60%
		self.closes_inside_ratio = params.get('closes_inside_ratio', 0.70)

		# Triggers
		self.rejection_wick_ratio = params.get('rejection_wick_ratio', 1.2)
		self.engulfing_min_atr = params.get('engulfing_min_atr', 0.25)
		self.momentum_body_atr = params.get('momentum_body_atr', 0.60)
		self.volume_mult_primary = params.get('volume_mult_primary', 1.5)
		self.volume_median_len = params.get('volume_median_len', 20)
		self.rejection_probe_atr = params.get('rejection_probe_atr', 0.10)

		# Stops/targets
		self.pivot_sl_buffer_atr = params.get('pivot_sl_buffer_atr', 0.40)
		self.box_sl_buffer_atr = params.get('box_sl_buffer_atr', 0.25)
		self.t1_profile_atr = params.get('t1_profile_atr', 0.75)
		self.scale_out_pct = params.get('scale_out_pct', 0.65)
		self.time_stop_bars = params.get('time_stop_bars', 12)

		# Attempts per level per session
		self.max_trades_per_session = params.get('max_trades_per_session', 3)
		self._level_attempts: Dict[str, Dict[str, int]] = {symbol: {} for symbol in symbols}
		self._losses_today = 0
		self._consec_losses = 0
		self.daily_r_stop = params.get('daily_r_stop', -2.0)
		self.max_consec_losses = params.get('max_consec_losses', 3)
		self.max_history_bars = max(400, self.ema200_len + 50)

	def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		for symbol in self.symbols:
			if symbol not in data or self._history[symbol].empty:
				continue
			df = self._history[symbol]
			if len(df) < self.max_history_bars:
				continue

			# Daily guardrails
			if self._consec_losses >= self.max_consec_losses:
				continue
			# Note: cumulative R tracking requires PnL tracking; placeholder stops at strategy layer

			tick = data[symbol]
			price = (tick['bid'] + tick['ask']) / 2
			spread = tick['ask'] - tick['bid']

			# Session windows
			if not self._in_session_window(df.index[-1]):
				continue

			# Compute pivots from prior day
			piv = self._compute_daily_pivots(df)
			if not piv:
				continue

			# Regime filters: 200-EMA and session VWAP alignment
			if not self._regime_ok(symbol, price):
				continue

			# ATR% band
			atr = self._calculate_atr(symbol, self.atr_len)
			if atr is None or price == 0:
				continue
			atr_pct = (atr / price) * 100
			if self._is_crypto(symbol):
				if not (self.atr_pct_min_crypto <= atr_pct <= self.atr_pct_max_crypto):
					continue
			else:
				if not (self.atr_pct_min_fx <= atr_pct <= self.atr_pct_max_fx):
					continue

			# Find strict consolidation around an active pivot level
			box = self._find_strict_box_around_pivot(symbol, df, piv, atr, price)
			if not box:
				continue

			# One try per level within session
			level_key = f"{box['level_tag']}:{df.index[-1].date()}"
			if self._level_attempts[symbol].get(level_key, 0) >= 1:
				continue

			# Planned SL for spread gate: use farther-of rule
			planned_sl = self._planned_sl_distance(box, atr)
			if planned_sl <= 0 or spread > self.spread_vs_sl_cap * planned_sl:
				continue

			# Primary trigger + confirmations
			primary = self._primary_trigger(symbol, df, box, atr, price)
			if not primary:
				continue
			confirms = self._confirmations(symbol, df, box, price)
			if confirms < 2:
				continue

			# Build order: enter next bar open (we use current close as proxy here)
			entry = df.iloc[-1]['close']
			long = primary['dir'] == 'LONG'
			sl = self._compute_sl(entry, box, atr, long)
			tp = self._compute_t1(symbol, box, piv, entry, long, atr)

			self._level_attempts[symbol][level_key] = 1
			order_type = 'BUY' if long else 'SELL'
			logging.info(f"PivotScalper {symbol}: {order_type} entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} level={box['level_tag']}")
			return {
				'symbol': symbol,
				'type': order_type,
				'sl': sl,
				'tp': tp,
				'entry_price': entry
			}

		return None

	# ===== Core checks =====
	def _regime_ok(self, symbol: str, price: float) -> bool:
		df = self._history[symbol]
		close = df['close']
		if len(close) < self.ema200_len:
			return False
		ema200 = close.ewm(span=self.ema200_len, adjust=False).mean().iloc[-1]
		vwap = self._session_vwap(df)
		if vwap is None:
			return False
		# Long regime
		if price >= ema200 and price >= vwap:
			return True
		# Short regime
		if price <= ema200 and price <= vwap:
			return True
		return False

	def _find_strict_box_around_pivot(self, symbol: str, df: pd.DataFrame, piv: Dict[str, float], atr: float, price: float) -> Optional[Dict[str, Any]]:
		# Choose active level: nearest among P,R1,S1 (allow R2/S2 in strong trend)
		levels = {
			'P': piv['P'], 'R1': piv['R1'], 'S1': piv['S1'], 'R2': piv['R2'], 'S2': piv['S2']
		}
		nearest_tag, nearest_level = min(levels.items(), key=lambda kv: abs(price - kv[1]))

		# Scan recent windows for 12–30 bars
		for bars in range(self.box_max_bars, self.box_min_bars - 1, -1):
			wnd = df.iloc[-bars:]
			mid = (wnd['high'].max() + wnd['low'].min()) / 2
			if abs(nearest_level - mid) > self.pivot_mid_distance_atr * atr:
				continue
			height = wnd['high'].max() - wnd['low'].min()
			cap = self.box_height_cap_gi if self._is_gi(symbol) else self.box_height_cap_fx
			if height > cap * atr:
				continue
			# BB inside Keltner ≥ 8 bars
			if not self._bb_inside_kc_consecutive(df, self.bb_in_kc_min_bars):
				continue
			# BB width caps
			bb = self._bb(df)
			if bb is None:
				continue
			bb_width = ((bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) / df['close'].iloc[-1])
			price_cap = self.bb_width_pct_price_cap_crypto if self._is_crypto(symbol) else self.bb_width_pct_price_cap
			if (bb['width_atr'].iloc[-1] > self.bb_width_pct_atr_cap * atr) or (bb_width > price_cap):
				continue
			# Closes containment ≥ 70%
			closes_in = ((wnd['close'] >= wnd['low'].min()) & (wnd['close'] <= wnd['high'].max())).mean()
			if closes_in < self.closes_inside_ratio:
				continue

			return {
				'level_tag': nearest_tag,
				'level_price': nearest_level,
				'box_high': wnd['high'].max(),
				'box_low': wnd['low'].min(),
				'height': height
			}
		return None

	def _primary_trigger(self, symbol: str, df: pd.DataFrame, box: Dict[str, Any], atr: float, price: float) -> Optional[Dict[str, Any]]:
		last = df.iloc[-1]
		prev = df.iloc[-2]
		box_high = box['box_high']
		box_low = box['box_low']
		level = box['level_price']

		# 1) Rejection-wick close-back-in
		probe_ok = (abs((last['low'] if price <= level else last['high']) - level) <= self.rejection_probe_atr * atr)
		wick_len = (last['high'] - last['close']) if price > level else (last['close'] - last['low'])
		body_len = abs(last['close'] - last['open'])
		if probe_ok and body_len > 0 and (wick_len / body_len) >= self.rejection_wick_ratio and \
		   ((price > level and last['close'] < box_high) or (price < level and last['close'] > box_low)):
			return {'dir': 'SHORT' if price > level else 'LONG'}

		# 2) Engulfing reversal
		if body_len >= self.engulfing_min_atr * atr and ((last['close'] > prev['open'] and last['open'] < prev['close']) or \
			(last['close'] < prev['open'] and last['open'] > prev['close'])):
			return {'dir': 'LONG' if last['close'] > prev['close'] else 'SHORT'}

		# 3) Momentum pop off pivot
		if body_len >= self.momentum_body_atr * atr and ((price >= level and last['close'] > level) or (price <= level and last['close'] < level)):
			return {'dir': 'LONG' if last['close'] > level else 'SHORT'}

		return None

	def _confirmations(self, symbol: str, df: pd.DataFrame, box: Dict[str, Any], price: float) -> int:
		cnt = 0
		last = df.iloc[-1]
		# Volume expansion
		vol_med = self._get_median_volume(symbol, self.volume_median_len)
		if vol_med is not None and last['tick_volume'] >= self.volume_mult_primary * vol_med:
			cnt += 1
		# RSI regime 50 cross at close
		rsi = self._calculate_rsi(symbol, self.rsi_len)
		if rsi is not None and ((price >= box['level_price'] and rsi > 50) or (price <= box['level_price'] and rsi < 50)):
			cnt += 1
		# VWAP reclaim/lose
		vwap = self._session_vwap(df)
		if vwap is not None and ((price > vwap and price >= box['level_price']) or (price < vwap and price <= box['level_price'])):
			cnt += 1
		# Liquidity sweep: take out prior swing in wrong direction by ≤0.15×ATR then close back in
		atr = self._calculate_atr(symbol, self.atr_len)
		if atr is not None:
			swing_lookback = df.iloc[-5:-1]
			if price >= box['level_price']:
				wrong_low = swing_lookback['low'].min()
				if (last['low'] <= wrong_low) and (abs(wrong_low - box['level_price']) <= 0.15 * atr) and (last['close'] > box['level_price']):
					cnt += 1
			else:
				wrong_high = swing_lookback['high'].max()
				if (last['high'] >= wrong_high) and (abs(wrong_high - box['level_price']) <= 0.15 * atr) and (last['close'] < box['level_price']):
					cnt += 1
		return cnt

	def _compute_sl(self, entry: float, box: Dict[str, Any], atr: float, long: bool) -> float:
		pivot_side = box['level_price'] - self.pivot_sl_buffer_atr * atr if long else box['level_price'] + self.pivot_sl_buffer_atr * atr
		box_side = box['box_low'] - self.box_sl_buffer_atr * atr if long else box['box_high'] + self.box_sl_buffer_atr * atr
		if long:
			return min(pivot_side, box_side)
		else:
			return max(pivot_side, box_side)

	def _compute_t1(self, symbol: str, box: Dict[str, Any], piv: Dict[str, float], entry: float, long: bool, atr: float) -> float:
		# If entering off Pivot → T1 = nearest R1/S1
		level = box['level_tag']
		if level == 'P':
			return piv['R1'] if long else piv['S1']
		# If entering off R1/S1 fade → Pivot or 0.75×ATR from entry (pick nearer)
		if level in ('R1', 'S1'):
			target_pivot = piv['P']
			alt = entry + (self.t1_profile_atr * atr if long else -self.t1_profile_atr * atr)
			return alt if abs(alt - entry) < abs(target_pivot - entry) else target_pivot
		# Default fallback
		return entry + (self.t1_profile_atr * atr if long else -self.t1_profile_atr * atr)

	def _planned_sl_distance(self, box: Dict[str, Any], atr: float) -> float:
		pivot_side = self.pivot_sl_buffer_atr * atr
		box_side = self.box_sl_buffer_atr * atr
		return max(pivot_side, box_side)

	# ===== Indicators & utils =====
	def _compute_daily_pivots(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
		if df.empty:
			return None
		last_ts = df.index[-1]
		yesterday = (last_ts - pd.Timedelta(days=1)).date()
		mask = df.index.date == yesterday
		day = df.loc[mask]
		if day.empty:
			# Fallback: previous 1440 minutes
			day = df.iloc[-1440:] if len(df) >= 1440 else df
			if day.empty:
				return None
		H = day['high'].max()
		L = day['low'].min()
		C = day['close'].iloc[-1]
		P = (H + L + C) / 3
		R1 = 2*P - L
		S1 = 2*P - H
		R2 = P + (H - L)
		S2 = P - (H - L)
		return {'P': P, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2}

	def _bb(self, df: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
		if len(df) < self.bb_len:
			return None
		close = df['close']
		sma = close.rolling(self.bb_len).mean()
		std = close.rolling(self.bb_len).std()
		upper = sma + self.bb_dev * std
		lower = sma - self.bb_dev * std
		width_atr = (upper - lower) / 2 # raw width proxy
		return {'upper': upper, 'lower': lower, 'width_atr': width_atr}

	def _kc(self, df: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
		if len(df) < self.bb_len:
			return None
		close = df['close']
		ema = close.ewm(span=self.bb_len, adjust=False).mean()
		tr1 = df['high'] - df['low']
		tr2 = (df['high'] - df['close'].shift()).abs()
		tr3 = (df['low'] - df['close'].shift()).abs()
		tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
		atr = tr.ewm(span=self.bb_len, adjust=False).mean()
		upper = ema + self.kc_mult * atr
		lower = ema - self.kc_mult * atr
		return {'upper': upper, 'lower': lower}

	def _bb_inside_kc_consecutive(self, df: pd.DataFrame, min_bars: int) -> bool:
		bb = self._bb(df)
		kc = self._kc(df)
		if not bb or not kc:
			return False
		up_ok = (bb['upper'] <= kc['upper']).iloc[-min_bars:]
		lo_ok = (bb['lower'] >= kc['lower']).iloc[-min_bars:]
		return bool((up_ok & lo_ok).all())

	def _session_vwap(self, df: pd.DataFrame) -> Optional[float]:
		# Use today's date cumulative VWAP approximation with tick_volume
		if df.empty:
			return None
		today = df.index[-1].date()
		day = df.loc[df.index.date == today]
		if day.empty:
			return None
		pv = day['close'] * day['tick_volume'].clip(lower=1)
		cum_pv = pv.cumsum()
		cum_v = day['tick_volume'].clip(lower=1).cumsum()
		return float((cum_pv / cum_v).iloc[-1])

	def _is_crypto(self, symbol: str) -> bool:
		return symbol.upper().startswith('BTC') or symbol.upper().startswith('ETH')

	def _is_gi(self, symbol: str) -> bool:
		return symbol.upper() in ('XAUUSD', 'XAGUSD', 'US100', 'NAS100', 'NAS100', 'US500', 'SPX500')

	def _in_session_window(self, ts: pd.Timestamp) -> bool:
		def parse(hhmm: str) -> dtime:
			try:
				h, m = map(int, hhmm.split(':'))
				return dtime(h, m)
			except Exception:
				return dtime(0, 0)
		lo = parse(self.session_london_open)
		lo_end = (datetime.combine(ts.date(), lo) + timedelta(hours=self.session_london_hours)).time()
		ny = parse(self.session_ny_open)
		ny_end = (datetime.combine(ts.date(), ny) + timedelta(hours=self.session_ny_hours)).time()
		t = ts.time()
		in_london = (t >= lo and t <= lo_end)
		in_ny = (t >= ny and t <= ny_end)
		if self.session_ny_last_hour:
			# Approx last 60m of NY session: assume 20:00-21:00 server local as placeholder
			return in_london or in_ny or (t >= dtime(20, 0) and t <= dtime(21, 0))
		return in_london or in_ny
