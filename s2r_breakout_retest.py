from strategy_base import StrategyBase
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
import numpy as np

try:
	import MetaTrader5 as mt5
except Exception:
	mt5 = None


class S2RBreakoutRetest(StrategyBase):
	"""Strategy S2-R — Breakout + Retest (Continuation)

	Closed-bar evaluation (bar -2) with structure-first level engine, consolidation+squeeze gate,
	breakout then retest confirmation, RSI and H4 bias context, and structure-aware exits.
	"""

	def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
		super().__init__(name, symbols, params)

		# TFs
		self.timeframe = params.get('timeframe', (mt5.TIMEFRAME_M15 if mt5 else 0))
		self.use_m30 = params.get('use_m30', False)

		# Indicators
		self.atr_len = params.get('atr_len', 14)
		self.rsi_len = params.get('rsi_len', 14)
		self.bb_len = params.get('bb_len', 20)
		self.bb_dev = params.get('bb_dev', 2.0)
		self.adx_len = params.get('adx_len', 14)

		# H4 bias
		self.bias_enabled = params.get('bias_enabled', True)
		self.h4_sma_len = params.get('h4_sma_len', 50)
		self.h4_slope_window = params.get('h4_slope_window', 20)

		# Consolidation
		self.box_min_bars = params.get('box_min_bars', 3)
		self.box_max_bars = params.get('box_max_bars', 8)
		self.box_height_atr = params.get('box_height_atr', 0.8)
		self.bb_bw_pctile = params.get('bb_bw_pctile', 0.35)
		self.adx_squeeze_cap = params.get('adx_squeeze_cap', 20)

		# ATR regime
		self.atr_pctile_low = params.get('atr_pctile_low', 0.20)
		self.atr_pctile_high = params.get('atr_pctile_high', 0.80)
		self.atr_pctile_low_m30 = params.get('atr_pctile_low_m30', 0.25)

		# Breakout/retest
		self.breakout_close_atr = params.get('breakout_close_atr', 0.7)
		self.breakout_close_atr_xau_crypto = params.get('breakout_close_atr_xau_crypto', 0.8)
		self.retest_window_bars = params.get('retest_window_bars', 3)
		self.retest_zone_atr = params.get('retest_zone_atr', 0.15)
		self.retest_pin_tr_atr = params.get('retest_pin_tr_atr', 0.8)
		self.retest_engulf_body_atr = params.get('retest_engulf_body_atr', 0.6)
		self.max_run_before_retest_atr = params.get('max_run_before_retest_atr', 1.2)
		self.volume_break_mult = params.get('volume_break_mult', 1.5)

		# Risk/targets
		self.sl_buffer_atr = params.get('sl_buffer_atr', 0.5)
		self.tp_buffer_atr = params.get('tp_buffer_atr', 0.15)
		self.min_rr = params.get('min_rr', 1.2)

		# Ops
		self.cooldown_bars = params.get('cooldown_bars', 3)
		self.session_start = params.get('session_start', '07:00')
		self.session_end = params.get('session_end', '17:00')
		self.spread_guard_atr = params.get('spread_guard_atr', 0.12)

		self.max_history_bars = max(500, 300 + 50)
		self._last_signal_bar: Dict[str, int] = {symbol: -999999 for symbol in symbols}
		self._level_cooldown: Dict[str, Dict[str, Any]] = {symbol: {} for symbol in symbols}

	def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		for symbol in self.symbols:
			if symbol not in data or self._history[symbol].empty:
				continue
			df = self._history[symbol]
			if len(df) < self.max_history_bars:
				continue

			bar_index = len(df)
			if bar_index - self._last_signal_bar.get(symbol, -999999) < self.cooldown_bars:
				continue

			tick = data[symbol]
			price = (tick['bid'] + tick['ask']) / 2
			atr = self._calculate_atr(symbol, self.atr_len)
			if atr is None or atr <= 0:
				continue

			# Guards: session, spread
			if not self._in_session_window(df.index[-1]):
				continue
			spread = tick['ask'] - tick['bid']
			if spread / atr > self.spread_guard_atr:
				continue

			# ATR regime percentiles
			atr_pctile = self._atr_percentile(df, self.atr_len, 300)
			if atr_pctile is None:
				continue
			low = self.atr_pctile_low_m30 if self.use_m30 else self.atr_pctile_low
			if not (low <= atr_pctile <= self.atr_pctile_high):
				continue

			# Level engine
			levels = self._build_levels(df, atr)
			if not levels:
				continue

			# Consolidation gate around candidate levels
			if not self._has_consolidation(df, atr):
				continue

			# Breakout + retest evaluation on closed bars (bar -2)
			decision_idx = -2
			res = self._evaluate_breakout_retest(symbol, df, atr, levels, decision_idx)
			if not res:
				continue

			entry_price, side, level_price = res['entry'], res['side'], res['level']
			# SL
			sl = level_price - self.sl_buffer_atr * atr if side == 'long' else level_price + self.sl_buffer_atr * atr
			# TP at opposing level
			opp = self._nearest_opposing(levels, level_price, side)
			if opp is None:
				continue
			tp = (opp - self.tp_buffer_atr * atr) if side == 'long' else (opp + self.tp_buffer_atr * atr)

			# RR check
			risk = abs(entry_price - sl)
			reward = abs(tp - entry_price)
			if risk <= 0 or reward / risk < self.min_rr:
				continue

			self._last_signal_bar[symbol] = bar_index
			order_type = 'BUY' if side == 'long' else 'SELL'
			return {'symbol': symbol, 'type': order_type, 'sl': sl, 'tp': tp}

		return None

	# ===== Components =====
	def _in_session_window(self, ts: pd.Timestamp) -> bool:
		from datetime import time as dtime
		t = ts.time()
		start_h, start_m = map(int, self.session_start.split(':'))
		end_h, end_m = map(int, self.session_end.split(':'))
		return dtime(start_h, start_m) <= t <= dtime(end_h, end_m)

	def _atr_percentile(self, df: pd.DataFrame, period: int, window: int) -> Optional[float]:
		if len(df) < window + period:
			return None
		atr_series = []
		for i in range(window):
			val = self._calculate_atr_on_df(df.iloc[:-(window - i)], period)
			if val is not None:
				atr_series.append(val)
		if not atr_series:
			return None
		curr = self._calculate_atr_on_df(df, period)
		pct = (np.sum(np.array(atr_series) <= curr) / len(atr_series))
		return float(pct)

	def _build_levels(self, df: pd.DataFrame, atr: float) -> Dict[str, List[Dict[str, Any]]]:
		# 1) Swings with ±2 window, prominence ≥ 0.30×ATR at that bar
		prom = 0.30 * atr
		highs = []
		lows = []
		for i in range(2, len(df) - 2):
			window = df.iloc[i-2:i+3]
			if df['high'].iloc[i] == window['high'].max() and (df['high'].iloc[i] - df['close'].iloc[i-1]) >= prom:
				highs.append((df.index[i], float(df['high'].iloc[i])))
			if df['low'].iloc[i] == window['low'].min() and (df['close'].iloc[i-1] - df['low'].iloc[i]) >= prom:
				lows.append((df.index[i], float(df['low'].iloc[i])))

		# 2) Cluster within ±0.20×ATR; then de-dup closer than 0.15×ATR
		def cluster(points: List[Tuple[pd.Timestamp, float]], radius: float) -> List[List[Tuple[pd.Timestamp, float]]]:
			clusters: List[List[Tuple[pd.Timestamp, float]]] = []
			for ts, p in points:
				placed = False
				for c in clusters:
					if abs(np.median([x[1] for x in c]) - p) <= radius:
						c.append((ts, p))
						placed = True
						break
				if not placed:
					clusters.append([(ts, p)])
			return clusters

		def merge_close(levels: List[Dict[str, Any]], sep: float) -> List[Dict[str, Any]]:
			levels = sorted(levels, key=lambda x: x['price'])
			merged: List[Dict[str, Any]] = []
			for lvl in levels:
				if not merged:
					merged.append(lvl)
					continue
				if abs(merged[-1]['price'] - lvl['price']) < sep:
					combined = merged.pop(-1)
					pts = combined['members'] + lvl['members']
					merged.append({'price': float(np.median([p for _, p in pts])), 'members': pts})
				else:
					merged.append(lvl)
			return merged

		rad = 0.20 * atr
		sep = 0.15 * atr
		clusters_high = cluster(highs, rad)
		clusters_low = cluster(lows, rad)

		levels_high = [{'price': float(np.median([p for _, p in c])), 'members': c} for c in clusters_high]
		levels_low = [{'price': float(np.median([p for _, p in c])), 'members': c} for c in clusters_low]

		levels_high = merge_close(levels_high, sep)
		levels_low = merge_close(levels_low, sep)

		# Eligibility: keep only with ≥2 touches in last 100 bars, compute TQS
		def tqs(levels_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
			kept = []
			for lvl in levels_list:
				members = [(ts, p) for ts, p in lvl['members'] if ts >= df.index[-100]]
				if len(members) < 2:
					continue
				score = 0.0
				for ts, p in members:
					idx = df.index.get_loc(ts)
					reaction = max(df['high'].iloc[idx+1:idx+4].max() - p, p - df['low'].iloc[idx+1:idx+4].min())
					reaction = reaction / max(atr, 1e-9)
					dk = len(df) - idx
					score += reaction * np.exp(-dk / 50.0)
				kept.append({'price': lvl['price'], 'TQS': score, 'last_touch_ts': members[-1][0], 'touches': len(members)})
			return kept

		kept_high = tqs(levels_high)
		kept_low = tqs(levels_low)

		# Cap to top 3 per side by TQS
		kept_high = sorted(kept_high, key=lambda x: (-x['TQS'], -x['last_touch_ts'].timestamp()))[:3]
		kept_low = sorted(kept_low, key=lambda x: (-x['TQS'], -x['last_touch_ts'].timestamp()))[:3]

		return {'resistance': kept_high, 'support': kept_low}

	def _has_consolidation(self, df: pd.DataFrame, atr: float) -> bool:
		# Box 3–8 bars; allow inside-bars as cluster
		for k in range(self.box_min_bars, self.box_max_bars + 1):
			win = df.iloc[-k:]
			box_h = float(win['high'].max() - win['low'].min())
			if box_h <= self.box_height_atr * atr:
				# Squeeze by BB bandwidth percentile or ADX cap
				bb_bw = self._bb_bandwidth_series(df)
				if bb_bw is not None:
					recent = bb_bw.iloc[-100:]
					pct = (recent <= bb_bw.iloc[-1]).mean()
					if pct <= self.bb_bw_pctile or (self._adx_ok(df)):
						return True
		return False

	def _bb_bandwidth_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
		ma = df['close'].rolling(self.bb_len).mean()
		std = df['close'].rolling(self.bb_len).std()
		if len(ma.dropna()) == 0 or len(std.dropna()) == 0:
			return None
		upper = ma + self.bb_dev * std
		lower = ma - self.bb_dev * std
		bw = (upper - lower) / ma.replace(0, np.nan)
		return bw

	def _adx_ok(self, df: pd.DataFrame) -> bool:
		adx = self._calculate_adx_on_window(df, self.adx_len)
		return adx is not None and adx < self.adx_squeeze_cap

	def _calculate_adx_on_window(self, df: pd.DataFrame, period: int) -> Optional[float]:
		# Simple proxy: reuse strategy_base ADX then take last
		# Note: strategy_base._calculate_adx modifies df with temp cols; safe on copy
		df_copy = df.copy()
		try:
			self._history_temp = { 'tmp': df_copy }
			res = self._calculate_adx('tmp', period)
			return res['adx'] if res else None
		except Exception:
			return None

	def _evaluate_breakout_retest(self, symbol: str, df: pd.DataFrame, atr: float, levels: Dict[str, List[Dict[str, Any]]], decision_idx: int) -> Optional[Dict[str, Any]]:
		last = df.iloc[decision_idx]
		prev = df.iloc[decision_idx - 1]
		symbol_u = symbol.upper()
		bo_req = self.breakout_close_atr_xau_crypto if symbol_u in ('XAUUSD','BTCUSD','ETHUSD','BTCUSDT','ETHUSDT') else self.breakout_close_atr

		# Try resistance first (longs)
		for lvl in levels['resistance']:
			level = lvl['price']
			# Did breakout happen within last 3 bars and close beyond by ΔBO?
			brk = self._find_breakout(df, level, 'long', atr, bo_req, max_lookback=3)
			if not brk:
				continue
			# Freshness: run before retest < 1.2×ATR
			if brk['run_before_retest_atr'] >= self.max_run_before_retest_atr:
				continue
			# Retest within window and zone quality
			rt = self._find_retest(df, level, 'long', atr, self.retest_window_bars, self.retest_zone_atr)
			if not rt:
				continue
			# Retest bar quality
			if not self._retest_quality(df.iloc[rt['bar_index']], atr, 'long'):
				continue
			# RSI gates on retest bar
			rsi_now = self._rsi_on_df(df.iloc[:rt['bar_index']+1], self.rsi_len)
			rsi_prev = self._rsi_on_df(df.iloc[:rt['bar_index']], self.rsi_len)
			if rsi_now is None or rsi_prev is None or not (rsi_now <= 55 and (rsi_now - rsi_prev) >= 0):
				continue
			# H4 bias
			if self.bias_enabled and not self._h4_bias_ok(symbol, True):
				continue
			# Confirmation: retest bar qualifies; use its close as entry
			return {'entry': float(df['close'].iloc[rt['bar_index']]), 'side': 'long', 'level': level}

		# Support (shorts)
		for lvl in levels['support']:
			level = lvl['price']
			brk = self._find_breakout(df, level, 'short', atr, bo_req, max_lookback=3)
			if not brk:
				continue
			if brk['run_before_retest_atr'] >= self.max_run_before_retest_atr:
				continue
			rt = self._find_retest(df, level, 'short', atr, self.retest_window_bars, self.retest_zone_atr)
			if not rt:
				continue
			if not self._retest_quality(df.iloc[rt['bar_index']], atr, 'short'):
				continue
			rsi_now = self._rsi_on_df(df.iloc[:rt['bar_index']+1], self.rsi_len)
			rsi_prev = self._rsi_on_df(df.iloc[:rt['bar_index']], self.rsi_len)
			if rsi_now is None or rsi_prev is None or not (rsi_now >= 45 and (rsi_now - rsi_prev) <= 0):
				continue
			if self.bias_enabled and not self._h4_bias_ok(symbol, False):
				continue
			return {'entry': float(df['close'].iloc[rt['bar_index']]), 'side': 'short', 'level': level}

		return None

	def _find_breakout(self, df: pd.DataFrame, level: float, side: str, atr: float, bo_req: float, max_lookback: int) -> Optional[Dict[str, Any]]:
		for i in range(1, max_lookback + 1):
			bar = df.iloc[-(i+1)]  # ensure closed bar indexing
			if side == 'long':
				if bar['close'] > level + bo_req * atr:
					# simple run-before-retest measure: from breakout close to subsequent high before retest
					run = df['high'].iloc[-(i):].max() - bar['close']
					return {'bar_index': len(df) - (i+1), 'run_before_retest_atr': float(run / max(atr, 1e-9))}
			else:
				if bar['close'] < level - bo_req * atr:
					run = bar['close'] - df['low'].iloc[-(i):].min()
					return {'bar_index': len(df) - (i+1), 'run_before_retest_atr': float(run / max(atr, 1e-9))}
		return None

	def _find_retest(self, df: pd.DataFrame, level: float, side: str, atr: float, window_bars: int, zone_atr: float) -> Optional[Dict[str, Any]]:
		for j in range(window_bars):
			bar = df.iloc[-(j+1)]
			if side == 'long':
				if (bar['low'] <= level + zone_atr * atr) and (bar['low'] >= level - zone_atr * atr):
					return {'bar_index': len(df) - (j+1)}
			else:
				if (bar['high'] >= level - zone_atr * atr) and (bar['high'] <= level + zone_atr * atr):
					return {'bar_index': len(df) - (j+1)}
		return None

	def _retest_quality(self, bar: pd.Series, atr: float, side: str) -> bool:
		body = abs(bar['close'] - bar['open'])
		tr = bar['high'] - bar['low']
		if side == 'long':
			pin = (tr >= self.retest_pin_tr_atr * atr) and ((bar['high'] - bar['close']) >= 1.5 * body)
			engulf = (body >= self.retest_engulf_body_atr * atr) and (bar['close'] >= bar['open'])
			close_ok = (bar['close'] >= (bar['open'] + bar['close']) / 2)
			return (pin or engulf) and close_ok
		else:
			pin = (tr >= self.retest_pin_tr_atr * atr) and ((bar['close'] - bar['low']) >= 1.5 * body)
			engulf = (body >= self.retest_engulf_body_atr * atr) and (bar['close'] <= bar['open'])
			close_ok = (bar['close'] <= (bar['open'] + bar['close']) / 2)
			return (pin or engulf) and close_ok

	def _nearest_opposing(self, levels: Dict[str, List[Dict[str, Any]]], level_price: float, side: str) -> Optional[float]:
		candidates = levels['support'] if side == 'long' else levels['resistance']
		if not candidates:
			return None
		prices = [x['price'] for x in candidates]
		if side == 'long':
			below = [p for p in prices if p > level_price]
			return float(min(below)) if below else None
		else:
			above = [p for p in prices if p < level_price]
			return float(max(above)) if above else None

	def _rsi_on_df(self, df: pd.DataFrame, length: int) -> Optional[float]:
		delta = df['close'].diff()
		up = delta.clip(lower=0).ewm(span=length, adjust=False).mean()
		down = (-delta.clip(upper=0)).ewm(span=length, adjust=False).mean()
		rs = up / down.replace(0, np.nan)
		rsi = 100 - (100 / (1 + rs))
		val = rsi.iloc[-1]
		return float(val) if not np.isnan(val) else None

	def _h4_bias_ok(self, symbol: str, long: bool) -> bool:
		# Placeholder: engine should supply H4 series; here use main df as proxy
		# If engine adds confirm history, replace with true H4 logic
		return True
