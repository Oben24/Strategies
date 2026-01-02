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


class MeanReversionScalper(StrategyBase):
	"""Mean-Reversion Scalping (Sparse Mode)"""

	def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
		super().__init__(name, symbols, params)

		# Indicators / params
		self.bb_len = params.get('bb_len', 20)
		self.bb_dev = params.get('bb_dev', 2.0)
		self.rsi_len = params.get('rsi_len', 14)
		self.rsi_micro = params.get('rsi_micro', 7)
		self.atr_len = params.get('atr_len', 14)
		self.adx_len = params.get('adx_len', 14)
		self.sma_mid_len = params.get('sma_mid_len', 20)
		self.vol_med_len = params.get('vol_med_len', 50)

		# Regime gates
		self.adx_cap_skip = params.get('adx_cap_skip', 18)
		self.adx_soft_cap = params.get('adx_soft_cap', 15) # 15–18 size −50% (not handled here; info only)
		self.sma_flat_eps = params.get('sma_flat_eps', 0.0007) # 0.07%
		self.atrpct_min = params.get('atrpct_min', 0.18)
		self.atrpct_max = params.get('atrpct_max', 0.45)
		self.bb_pctl_len = params.get('bb_pctl_len', 200)
		self.bb_pctl_thresh = params.get('bb_pctl_thresh', 0.30) # 30th percentile
		self.bb_roc_len = params.get('bb_roc_len', 10)
		self.bb_roc_cap = params.get('bb_roc_cap', 0.10) # < +10%
		self.vwap_tether_atr_mult = params.get('vwap_tether_atr_mult', 1.0)
		self.rsi14_med_window = params.get('rsi14_med_window', 10)
		self.rsi14_med_low = params.get('rsi14_med_low', 42)
		self.rsi14_med_high = params.get('rsi14_med_high', 58)

		# Consolidation box
		self.box_min_bars = params.get('box_min_bars', 25)
		self.range20_atr_cap = params.get('range20_atr_cap', 1.0)
		self.min_outer_touches = params.get('min_outer_touches', 4)
		self.no_displacement_atr = params.get('no_displacement_atr', 2.0)
		self.touch_vol_mult = params.get('touch_vol_mult', 1.6)
		self.reentry_vol_mult = params.get('reentry_vol_mult', 1.1)

		# Entry specifics
		self.bband_reentry_lb = params.get('bband_reentry_lb', -0.15) # %B flip threshold
		self.rsi7_cross_from = params.get('rsi7_cross_from', 25)
		self.rsi7_cross_to = params.get('rsi7_cross_to', 35)
		self.rsi14_cross_from = params.get('rsi14_cross_from', 30)
		self.rsi14_cross_to = params.get('rsi14_cross_to', 35)
		self.wick_min_atr = params.get('wick_min_atr', 0.4)
		self.sweep_min_atr = params.get('sweep_min_atr', 0.2)
		self.limit_fill_min = params.get('limit_fill_min', 0.33)
		self.limit_fill_max = params.get('limit_fill_max', 0.50)
		self.limit_fill_wait_bars = params.get('limit_fill_wait_bars', 2)

		# Risk / management
		self.stop_min_atr = params.get('stop_min_atr', 0.35)
		self.stop_max_atr = params.get('stop_max_atr', 0.50)
		self.tp1_scale_pct = params.get('tp1_scale_pct', 0.85) # 80–90% range; use 85%
		self.time_stop_min = params.get('time_stop_min', 8)
		self.time_stop_max = params.get('time_stop_max', 10)

		# Microstructure vetoes
		self.spread_vs_sl_cap = params.get('spread_vs_sl_cap', 0.25)
		self.slippage_atr_cap = params.get('slippage_atr_cap', 0.15) # requires external metric
		self.gap_veto_atr = params.get('gap_veto_atr', 0.3)

		# Multi-TF alignment
		self.mt_tf = params.get('mt_tf', (mt5.TIMEFRAME_M15 if mt5 else 0))
		self.mt_sma_len = params.get('mt_sma_len', 20)
		self.mt_flat_eps = params.get('mt_flat_eps', 0.0005) # 0.05%

		# Cooldowns / session
		self.cooldown_after_tp1 = params.get('cooldown_after_tp1', 3)
		self.cooldown_after_loss = params.get('cooldown_after_loss', 10)
		self.max_trades_per_hour = params.get('max_trades_per_hour', 2)
		self.news_embargo_min = params.get('news_embargo_min', 20)

		self._pending_limit: Dict[str, Optional[Dict[str, Any]]]= {symbol: None for symbol in symbols}
		self._cooldown_dir: Dict[str, Dict[str, int]] = {symbol: {'LONG':0, 'SHORT':0} for symbol in symbols}
		self.max_history_bars = max(450, self.bb_pctl_len + 50)

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
			atr = self._calculate_atr(symbol, self.atr_len)
			if atr is None or price == 0:
				continue

			# Cooldowns
			if self._cooldown_dir[symbol]['LONG'] > 0:
				self._cooldown_dir[symbol]['LONG'] -= 1
			if self._cooldown_dir[symbol]['SHORT'] > 0:
				self._cooldown_dir[symbol]['SHORT'] -= 1

			# Regime filters
			reg_ok, soft_ok = self._regime_filters(symbol, price, atr)
			if not reg_ok:
				continue

			# Consolidation box detection
			box = self._detect_box(symbol, df, atr)
			if not box:
				continue

			# Pending limit logic: if exists, check fill window
			pending = self._pending_limit.get(symbol)
			if pending:
				if df.index[-1] > pending['expire_ts']:
					self._pending_limit[symbol] = None
				else:
					# Check if price revisited the retrace zone
					lo = min(df['low'].iloc[-1], df['low'].iloc[-2]) if len(df) >= 2 else df['low'].iloc[-1]
					hi = max(df['high'].iloc[-1], df['high'].iloc[-2]) if len(df) >= 2 else df['high'].iloc[-1]
					if (pending['long'] and lo <= pending['entry_price']) or ((not pending['long']) and hi >= pending['entry_price']):
						entry = pending['entry_price']
						long = pending['long']
						self._pending_limit[symbol] = None
						sl = self._compute_stop(symbol, box, atr, long)
						if sl is None:
							continue
						r = (entry - sl) if long else (sl - entry)
						if r <= 0:
							continue
						tp1 = self._compute_tp1(symbol, df, box, entry, long)
						if tp1 is None:
							continue
						# Spread vs SL cap
						if spread > self.spread_vs_sl_cap * abs(r):
							continue
						return {
							'symbol': symbol,
							'type': 'BUY' if long else 'SELL',
							'sl': sl,
							'tp': tp1,
							'entry_price': entry
						}

			# Entry evaluation
			signal = self._entry_signal(symbol, df, box, atr)
			if not signal:
				continue

			# Compute retrace (limit-on-pullback)
			retr = self._retrace_entry_price(df, signal)
			if retr is None:
				# Not in retrace now; set pending for next up to 2 bars
				self._pending_limit[symbol] = {
					'entry_price': signal['retrace_price'],
					'expire_ts': df.index[-1] + pd.Timedelta(minutes=self.limit_fill_wait_bars),
					'long': signal['long']
				}
				continue

			# If retrace present now, build order immediately
			entry = retr
			long = signal['long']
			sl = self._compute_stop(symbol, box, atr, long)
			if sl is None:
				continue
			r = (entry - sl) if long else (sl - entry)
			if r <= 0:
				continue
			tp1 = self._compute_tp1(symbol, df, box, entry, long)
			if tp1 is None:
				continue
			if spread > self.spread_vs_sl_cap * abs(r):
				continue

			return {
				'symbol': symbol,
				'type': 'BUY' if long else 'SELL',
				'sl': sl,
				'tp': tp1,
				'entry_price': entry
			}
		return None

	# ===== Regime and box detection =====
	def _regime_filters(self, symbol: str, price: float, atr: float) -> Tuple[bool, bool]:
		df = self._history[symbol]
		# 1) ADX <= 15 (15–18 soft)
		adx = self._calculate_adx(symbol, self.adx_len)
		if adx is None:
			return False, False
		if adx['adx'] > self.adx_cap_skip:
			return False, False
		soft_ok = adx['adx'] <= self.adx_soft_cap
		# 2) SMA20 flatness
		sma20_now = self._calculate_sma(symbol, self.sma_mid_len)
		sma20_prev = self._get_sma_value_n_bars_ago(symbol, self.sma_mid_len, 10)
		if sma20_now is None or sma20_prev is None or price == 0:
			return False, soft_ok
		if abs(sma20_now - sma20_prev) / price >= self.sma_flat_eps:
			return False, soft_ok
		# 3) ATR% band
		atrpct = (atr / price) * 100
		if not (self.atrpct_min <= atrpct <= self.atrpct_max):
			return False, soft_ok
		# 4) BBWidth percentile
		bb_up, bb_lo, bb_w = self._bb_series(df)
		if bb_w is None:
			return False, soft_ok
		pctl = self._percentile_of_last(bb_w, self.bb_pctl_len)
		if pctl is None or pctl > self.bb_pctl_thresh:
			# soft failure
			pass
		# 5) BBWidth acceleration 10-bar ROC
		bb_roc = self._roc(bb_w, self.bb_roc_len)
		if bb_roc is not None and bb_roc > self.bb_roc_cap:
			# soft failure
			pass
		# 6) VWAP tether
		vwap = self._session_vwap(df)
		if vwap is None or abs(price - vwap) > self.vwap_tether_atr_mult * atr:
			# soft
			pass
		# 7) RSI14 rolling median in [42,58]
		rsi14_vals = self._rolling_rsi(symbol, self.rsi_len, self.rsi14_med_window)
		if rsi14_vals is None:
			return False, soft_ok
		med = np.median(rsi14_vals)
		if not (self.rsi14_med_low <= med <= self.rsi14_med_high):
			# soft
			pass
		return True, soft_ok

	def _detect_box(self, symbol: str, df: pd.DataFrame, atr: float) -> Optional[Dict[str, Any]]:
		# Min bars ≥ 25; range height last-20 HL ≤ 1.0×ATR(14)
		if len(df) < max(self.box_min_bars, 25) + 5:
			return None
		box_df = df.tail(max(self.box_min_bars, 25))
		range20 = df.tail(20)
		range_hl = range20['high'].max() - range20['low'].min()
		if range_hl > self.range20_atr_cap * atr:
			return None
		# touches ≥ 4 on outer zones
		outer_top = box_df['high'] >= (box_df['high'].max() - 0.1 * atr)
		outer_bot = box_df['low'] <= (box_df['low'].min() + 0.1 * atr)
		if (outer_top.sum() + outer_bot.sum()) < self.min_outer_touches:
			return None
		# no displacement: no close beyond ±2.0×ATR from box mid
		mid = (box_df['high'].max() + box_df['low'].min()) / 2
		if (abs(box_df['close'] - mid) > self.no_displacement_atr * atr).any():
			return None
		# volume pattern at the last touch vs re-entry
		vol_med = self._get_median_volume(symbol, self.vol_med_len)
		if vol_med is None:
			return None
		touch_bar = box_df.iloc[-1]
		reentry_bar = df.iloc[-1]
		if not (touch_bar['tick_volume'] >= self.touch_vol_mult * vol_med and reentry_bar['tick_volume'] <= self.reentry_vol_mult * vol_med):
			# soft: can be waived via quality pack C
			pass
		return {
			'box_high': box_df['high'].max(),
			'box_low': box_df['low'].min(),
			'height': box_df['high'].max() - box_df['low'].min(),
			'mid': mid
		}

	# ===== Entry construction =====
	def _entry_signal(self, symbol: str, df: pd.DataFrame, box: Dict[str, Any], atr: float) -> Optional[Dict[str, Any]]:
		# Long re-entry model (mirror for short)
		u, l, _ = self._bb_series(df)
		if u is None:
			return None
		prev = df.iloc[-2]
		last = df.iloc[-1]
		# Band re-entry
		percent_b_prev = self._percent_b_series(df).iloc[-2]
		percent_b_last = self._percent_b_series(df).iloc[-1]
		long_ok = (prev['close'] < l.iloc[-2] and last['close'] > l.iloc[-1] and percent_b_prev <= self.bband_reentry_lb and percent_b_last > self.bband_reentry_lb)
		short_ok = (prev['close'] > u.iloc[-2] and last['close'] < u.iloc[-1] and percent_b_prev >= (1 - self.bband_reentry_lb) and percent_b_last < (1 - self.bband_reentry_lb))
		# RSI micro cross
		rsi7_series = self._rsi_series(symbol, self.rsi_micro)
		rsi14_series = self._rsi_series(symbol, self.rsi_len)
		micro_long = (self._cross_from_to(rsi7_series, self.rsi7_cross_from, self.rsi7_cross_to, 2) or self._cross_from_to(rsi14_series, self.rsi14_cross_from, self.rsi14_cross_to, 2))
		micro_short = (self._cross_from_to(rsi7_series, 100-self.rsi7_cross_from, 100-self.rsi7_cross_to, 2) or self._cross_from_to(rsi14_series, 100-self.rsi14_cross_from, 100-self.rsi14_cross_to, 2))
		# Candle anatomy (reversal bar)
		rev_long = self._reversal_wick(last, atr, True)
		rev_short = self._reversal_wick(last, atr, False)
		# Sweep & reclaim
		sweep_long = self._sweep_reclaim(df, atr, True)
		sweep_short = self._sweep_reclaim(df, atr, False)

		if long_ok and micro_long and rev_long and sweep_long:
			retr_price = self._retr_price(last, True)
			return {'long': True, 'retrace_price': retr_price, 'signal_bar': last}
		if short_ok and micro_short and rev_short and sweep_short:
			retr_price = self._retr_price(last, False)
			return {'long': False, 'retrace_price': retr_price, 'signal_bar': last}
		return None

	def _retrace_entry_price(self, df: pd.DataFrame, signal: Dict[str, Any]) -> Optional[float]:
		last = df.iloc[-1]
		bar = signal['signal_bar']
		low, high, open_, close = bar['low'], bar['high'], bar['open'], bar['close']
		body_mid = (open_ + close) / 2
		if signal['long']:
			zone_low = close - self.limit_fill_max * (close - low)
			zone_high = close - self.limit_fill_min * (close - low)
			# If current bar trades into zone, return current close as proxy
			if last['low'] <= zone_high:
				return max(last['close'], zone_low)
			return None
		else:
			zone_high = close + self.limit_fill_max * (high - close)
			zone_low = close + self.limit_fill_min * (high - close)
			if last['high'] >= zone_low:
				return min(last['close'], zone_high)
			return None

	def _compute_stop(self, symbol: str, box: Dict[str, Any], atr: float, long: bool) -> Optional[float]:
		# farther of (0.35–0.50×ATR beyond extreme) or beyond last swing
		last = self._history[symbol].iloc[-1]
		if long:
			extreme = last['low']
			stop1 = extreme - self.stop_max_atr * atr
			stop2 = box['box_low'] - self.stop_min_atr * atr
			return min(stop1, stop2)
		else:
			extreme = last['high']
			stop1 = extreme + self.stop_max_atr * atr
			stop2 = box['box_high'] + self.stop_min_atr * atr
			return max(stop1, stop2)

	def _compute_tp1(self, symbol: str, df: pd.DataFrame, box: Dict[str, Any], entry: float, long: bool) -> Optional[float]:
		sma20 = self._calculate_sma(symbol, self.sma_mid_len)
		if sma20 is None:
			return None
		# If slope <= 0.05%, take 80–90% scale; else take 100%
		sma20_prev = self._get_sma_value_n_bars_ago(symbol, self.sma_mid_len, 5)
		if sma20_prev is None or entry == 0:
			return sma20
		slope = abs(sma20 - sma20_prev) / entry
		return sma20

	# ===== Utilities =====
	def _percent_b_series(self, df: pd.DataFrame) -> pd.Series:
		up, lo, _ = self._bb_series(df)
		if up is None:
			return pd.Series(index=df.index, dtype=float)
		last_close = df['close']
		width = up - lo
		pb = (last_close - lo) / width.replace(0, np.nan)
		return pb

	def _bb_series(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
		if len(df) < self.bb_len:
			return None, None, None
		close = df['close']
		sma = close.rolling(self.bb_len).mean()
		std = close.rolling(self.bb_len).std()
		upper = sma + self.bb_dev * std
		lower = sma - self.bb_dev * std
		width = upper - lower
		return upper, lower, width

	def _roc(self, series: pd.Series, length: int) -> Optional[float]:
		if len(series) < length + 1:
			return None
		prev = series.iloc[-length-1]
		curr = series.iloc[-1]
		if prev == 0 or pd.isna(prev) or pd.isna(curr):
			return None
		return (curr - prev) / abs(prev)

	def _percentile_of_last(self, series: pd.Series, window_len: int) -> Optional[float]:
		window = series.tail(window_len).dropna()
		if len(window) < window_len:
			return None
		val = series.iloc[-1]
		return np.mean(window <= val)

	def _rolling_rsi(self, symbol: str, length: int, window: int) -> Optional[List[float]]:
		if len(self._history[symbol]) < length + window:
			return None
		vals = []
		for i in range(window):
			val = self._calculate_rsi_on_df(self._history[symbol].iloc[:-(window - i)], length, 'close')
			if val is None:
				return None
			vals.append(val)
		return vals

	def _session_vwap(self, df: pd.DataFrame) -> Optional[float]:
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

	def _reversal_wick(self, bar: pd.Series, atr: float, long: bool) -> bool:
		body = abs(bar['close'] - bar['open'])
		if body <= 0:
			return False
		wick = (bar['close'] - bar['low']) if long else (bar['high'] - bar['close'])
		return wick >= self.wick_min_atr * atr

	def _sweep_reclaim(self, df: pd.DataFrame, atr: float, long: bool) -> bool:
		recent = df.iloc[-6:-1]
		if recent.empty:
			return False
		if long:
			sw_low = recent['low'].min()
			last = df.iloc[-1]
			return (last['low'] <= sw_low - self.sweep_min_atr * atr) and (last['close'] > df['close'].ewm(span=self.bb_len, adjust=False).mean().iloc[-1] - self.bb_dev * df['close'].rolling(self.bb_len).std().iloc[-1])
		else:
			sw_high = recent['high'].max()
			last = df.iloc[-1]
			return (last['high'] >= sw_high + self.sweep_min_atr * atr) and (last['close'] < df['close'].ewm(span=self.bb_len, adjust=False).mean().iloc[-1] + self.bb_dev * df['close'].rolling(self.bb_len).std().iloc[-1])

	def _retr_price(self, bar: pd.Series, long: bool) -> float:
		# Return mid of 33–50% retrace
		if long:
			return bar['close'] - 0.415 * (bar['close'] - bar['low'])
		else:
			return bar['close'] + 0.415 * (bar['high'] - bar['close'])
