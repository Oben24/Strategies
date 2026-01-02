from strategy_base import StrategyBase
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
	import MetaTrader5 as mt5
except Exception:
	mt5 = None


class NewsBasedScalper(StrategyBase):
	"""Based Scalping Strategy (Bot-Ready Final Version)

	Trade only high-impact news events using:
	- Module A: Pre-News Straddle
	- Module B: Post-News Continuation
	- Module C: Fade (rare)

	This implementation requires event context in params:
	- event_time (ISO string or epoch seconds)
	- event_type (e.g., 'NFP', 'CPI', 'RATE')
	- surprise (float, standardized units per event)
	- expected_surprise (bool) [optional]
	- 2h ATR baseline will be computed from 1m series available to the strategy
	"""

	def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
		super().__init__(name, symbols, params)
		# Timeframes (engine feeds history for primary timeframe)
		self.timeframe = params.get('timeframe', (mt5.TIMEFRAME_M1 if mt5 else 0))
		self.aux_tf = params.get('aux_tf', (mt5.TIMEFRAME_M5 if mt5 else 0))

		# Event context
		evt = params.get('event_time')
		self.event_time = self._parse_event_time(evt)
		self.event_type = (params.get('event_type') or '').upper()
		self.surprise = params.get('surprise', None)
		self.expected_surprise = params.get('expected_surprise', False)

		# Surprise thresholds
		self.surprise_thresholds = params.get('surprise_thresholds', {
			'NFP': 75.0,
			'CPI': 0.2,
			'CORE CPI': 0.2,
			'RATE': 0.01, # unexpected change ~1pp; use language flag
			'FOMC': 0.01
		})

		# Filters/gates
		self.spread_guard_mult = params.get('spread_guard_mult', 0.15) # × ATR1m
		self.slippage_guard_mult = params.get('slippage_guard_mult', 0.4) # × ATR1m
		self.latency_ms_cap = params.get('latency_ms_cap', 5)
		self.liquidity_rel_cap = params.get('liquidity_rel_cap', 1.0) # tick vol last 30s ≤ median/hr

		# Straddle settings
		self.straddle_offset_atr = params.get('straddle_offset_atr', 0.8)
		self.straddle_place_sec = params.get('straddle_place_sec', 90)
		self.straddle_cancel_after_sec = params.get('straddle_cancel_after_sec', 15)
		self.sl_atr_mult = params.get('sl_atr_mult', 1.2)
		self.tp1_atr_mult = params.get('tp1_atr_mult', 1.0)
		self.tp2_atr_mult = params.get('tp2_atr_mult', 2.0)
		self.trailing_tp2_chand_atr = params.get('trailing_tp2_chand_atr', 2.5)

		# Continuation settings
		self.impulse_min_atr = params.get('impulse_min_atr', 1.5)
		self.cont_pivot_sl_atr = params.get('cont_pivot_sl_atr', 0.5)
		self.cont_tp_atr_min = params.get('cont_tp_atr_min', 1.5)
		self.cont_tp_atr_max = params.get('cont_tp_atr_max', 2.5)

		# Fade settings
		self.fade_spike_min_atr = params.get('fade_spike_min_atr', 2.2)
		self.fade_sl_extra_atr = params.get('fade_sl_extra_atr', 0.6)

		# Global frequency
		self.per_event_max_trades = params.get('per_event_max_trades', 1)
		self.daily_max_trades = params.get('daily_max_trades', 1)
		self.monthly_max_trades = params.get('monthly_max_trades', 5)

		# Internal state
		self._trades_today = 0
		self._trades_month = 0
		self._trades_this_event = 0
		self._last_event_day = None
		self._box: Dict[str, Optional[Dict[str, Any]]] = {symbol: None for symbol in symbols}
		self._armed_straddle: Dict[str, bool] = {symbol: False for symbol in symbols}
		self._armed_continuation: Dict[str, bool] = {symbol: False for symbol in symbols}
		self._armed_fade: Dict[str, bool] = {symbol: False for symbol in symbols}
		self._straddle_side_used: Dict[str, Optional[str]] = {symbol: None for symbol in symbols}
		self.max_history_bars = max(300, 200 + 120)

	def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		if self.event_time is None or self.surprise is None:
			return None

		day = datetime.utcnow().date()
		if self._last_event_day != day:
			self._trades_today = 0
			self._last_event_day = day

		if self._trades_today >= self.daily_max_trades or self._trades_month >= self.monthly_max_trades:
			return None

		for symbol in self.symbols:
			if symbol not in data or self._history[symbol].empty:
				continue
			df = self._history[symbol]
			if len(df) < self.max_history_bars:
				continue

			# Surprise gate
			if not self._surprise_ok():
				continue

			now = df.index[-1]
			tick = data[symbol]
			price = (tick['bid'] + tick['ask']) / 2
			spread = tick['ask'] - tick['bid']

			# Pre-news consolidation filter in [T-20m, T-2m]
			if self.event_time - timedelta(minutes=20) <= now <= self.event_time - timedelta(minutes=2):
				box = self._pre_news_consolidation(symbol)
				if box:
					self._box[symbol] = box
					logging.debug(f"{symbol} pre-news box set: {box}")
					# Arm straddle window if close to T
					if self.event_time - timedelta(seconds=self.straddle_place_sec) <= now <= self.event_time:
						if self._exec_filters(symbol, spread):
							self._armed_straddle[symbol] = True
							self._armed_continuation[symbol] = True
							self._armed_fade[symbol] = True

			# Straddle trigger within T to T+15s
			if self._armed_straddle[symbol] and self.event_time <= now <= self.event_time + timedelta(seconds=self.straddle_cancel_after_sec):
				box = self._box.get(symbol)
				if not box:
					continue
				atr1 = self._atr1m(symbol)
				if atr1 is None:
					continue
				up = box['high'] + self.straddle_offset_atr * atr1
				dn = box['low'] - self.straddle_offset_atr * atr1
				last = df.iloc[-1]
				# Execute market immediately upon cross
				if last['close'] >= up and self._straddle_side_used[symbol] is None:
					order = self._build_straddle_order(symbol, True, last['close'], atr1, box)
					self._armed_straddle[symbol] = False
					self._straddle_side_used[symbol] = 'BUY'
					return order
				elif last['close'] <= dn and self._straddle_side_used[symbol] is None:
					order = self._build_straddle_order(symbol, False, last['close'], atr1, box)
					self._armed_straddle[symbol] = False
					self._straddle_side_used[symbol] = 'SELL'
					return order

			# Continuation after T, up to T+30m
			if self._armed_continuation[symbol] and self.event_time <= now <= self.event_time + timedelta(minutes=30):
				order = self._post_news_continuation(symbol)
				if order:
					self._armed_continuation[symbol] = False
					return order

			# Fade - rare
			if self._armed_fade[symbol] and self.event_time <= now <= self.event_time + timedelta(minutes=10):
				order = self._fade_entry(symbol)
				if order:
					self._armed_fade[symbol] = False
					return order

		return None

	# ===== Gates & helpers =====
	def _parse_event_time(self, evt: Any) -> Optional[datetime]:
		if evt is None:
			return None
		if isinstance(evt, (int, float)):
			try:
				return datetime.utcfromtimestamp(evt)
			except Exception:
				return None
		if isinstance(evt, str):
			try:
				return datetime.fromisoformat(evt.replace('Z','+00:00')).replace(tzinfo=None)
			except Exception:
				return None
		return None

	def _surprise_ok(self) -> bool:
		if self.event_type in ('NFP', 'PAYROLLS'):
			return abs(self.surprise) >= self.surprise_thresholds['NFP']
		if self.event_type in ('CPI', 'CORE CPI'):
			return abs(self.surprise) >= self.surprise_thresholds['CPI']
		if self.event_type in ('RATE', 'FOMC', 'ECB', 'BOE'):
			# Expect bool flag expected_surprise or any rate delta provided externally
			return bool(self.expected_surprise) or abs(self.surprise) >= self.surprise_thresholds['RATE']
		# Allow optional lower-priority events only if specifically enabled
		return False

	def _atr1m(self, symbol: str) -> Optional[float]:
		return self._calculate_atr(symbol, self.atr_len)

	def _atr1m_baseline_2h(self, symbol: str) -> Optional[float]:
		df = self._history[symbol]
		if len(df) < 120:
			return None
		atr_series = []
		for i in range(120):
			val = self._calculate_atr_on_df(df.iloc[:-(120-i)], self.atr_len)
			if val is None:
				return None
			atr_series.append(val)
		return float(np.mean(atr_series)) if atr_series else None

	def _exec_filters(self, symbol: str, spread: float) -> bool:
		atr1 = self._atr1m(symbol)
		base = self._atr1m_baseline_2h(symbol)
		if atr1 is None or base is None:
			return False
		# ATR readiness
		if atr1 / base > 0.70:
			return False
		# Spread guard around news
		if spread > self.spread_guard_mult * atr1:
			return False
		# Placeholders: slippage/latency/liquidity must be integrated via external metrics
		return True

	def _pre_news_consolidation(self, symbol: str) -> Optional[Dict[str, Any]]:
		df = self._history[symbol]
		# Use last 20 minutes window (assuming 1m)
		wnd = df.tail(20)
		if len(wnd) < 18:
			return None
		# Zone via simple max/min
		z_high = wnd['high'].max()
		z_low = wnd['low'].min()
		mid = (z_high + z_low) / 2
		close_inside = ((wnd['close'] >= z_low) & (wnd['close'] <= z_high)).sum()
		if close_inside < 12:
			return None
		# Range width vs ATR5m(14) approximated by 1m ATR scaled
		atr1 = self._atr1m(symbol)
		if atr1 is None:
			return None
		if (z_high - z_low) > 0.35 * atr1:
			return None
		# Volatility suppression: mean ATR1m <= 0.60 × 2h baseline
		base = self._atr1m_baseline_2h(symbol)
		if base is None or np.mean([self._calculate_atr_on_df(df.iloc[:-(20 - i)], self.atr_len) for i in range(20)]) > 0.60 * base:
			return None
		# Drift filter: net price change <= 0.25 × ATR1m
		if abs(wnd['close'].iloc[-1] - wnd['close'].iloc[0]) > 0.25 * atr1:
			return None
		# Wick filter
		wicks = np.maximum(wnd['high'] - np.maximum(wnd['open'], wnd['close']), np.maximum(np.minimum(wnd['open'], wnd['close']) - wnd['low'], 0))
		if wicks.max() > 0.50 * atr1:
			return None
		# Max candle body size
		bodies = (wnd['close'] - wnd['open']).abs()
		if bodies.max() > 0.40 * atr1:
			return None
		return {'high': float(z_high), 'low': float(z_low), 'mid': float(mid)}

	def _build_straddle_order(self, symbol: str, long: bool, entry: float, atr1: float, box: Dict[str, Any]) -> Dict[str, Any]:
		# SL: max(1.2×ATR1m, just beyond box)
		if long:
			sl1 = entry - self.sl_atr_mult * atr1
			sl2 = box['low'] - 0.1 * atr1
			sl = min(sl1, sl2)
			tp = entry + self.tp1_atr_mult * atr1
		else:
			sl1 = entry + self.sl_atr_mult * atr1
			sl2 = box['high'] + 0.1 * atr1
			sl = max(sl1, sl2)
			tp = entry - self.tp1_atr_mult * atr1
		return {
			'symbol': symbol,
			'type': 'BUY' if long else 'SELL',
			'sl': sl,
			'tp': tp,
			'entry_price': entry
		}

	def _post_news_continuation(self, symbol: str) -> Optional[Dict[str, Any]]:
		df = self._history[symbol]
		last = df.iloc[-1]
		atr1 = self._atr1m(symbol)
		if atr1 is None:
			return None
		# Initial impulse magnitude measured over last 3 bars
		imp = df['high'].iloc[-3:].max() - df['low'].iloc[-3:].min()
		if imp < self.impulse_min_atr * atr1:
			return None
		# Pullback pivot: use last swing high/low in last 5 bars
		recent = df.iloc[-6:-1]
		pivot_long = recent['high'].max()
		pivot_short = recent['low'].min()
		# Breakout above/below pivot
		if last['close'] > pivot_long:
			entry = last['close']
			sl = pivot_long - self.cont_pivot_sl_atr * atr1
			tp = entry + self.cont_tp_atr_min * atr1
			return {'symbol': symbol, 'type': 'BUY', 'sl': sl, 'tp': tp, 'entry_price': entry}
		elif last['close'] < pivot_short:
			entry = last['close']
			sl = pivot_short + self.cont_pivot_sl_atr * atr1
			tp = entry - self.cont_tp_atr_min * atr1
			return {'symbol': symbol, 'type': 'SELL', 'sl': sl, 'tp': tp, 'entry_price': entry}
		return None

	def _fade_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
		df = self._history[symbol]
		atr1 = self._atr1m(symbol)
		if atr1 is None:
			return None
		# Spike in ≤30s approximated by last 2 bars span
		span = df['high'].iloc[-2:].max() - df['low'].iloc[-2:].min()
		if span < self.fade_spike_min_atr * atr1:
			return None
		last = df.iloc[-1]
		prev = df.iloc[-2]
		# Simple contradiction: body direction opposite to prior bar direction
		contradict = ((last['close'] < last['open']) and (prev['close'] > prev['open'])) or ((last['close'] > last['open']) and (prev['close'] < prev['open']))
		if not contradict:
			return None
		# Retrace limit at ~38.2%–50% of impulse
		impulse = abs(prev['close'] - last['close']) + 1e-9
		retr = 0.415 * impulse
		if last['close'] > last['open']:
			entry = last['close'] - retr
			sl = df['high'].iloc[-2:].max() + self.fade_sl_extra_atr * atr1
			tp = self._session_vwap(df) or (last['close'] - 0.618 * impulse)
			return {'symbol': symbol, 'type': 'SELL', 'sl': sl, 'tp': tp, 'entry_price': entry}
		else:
			entry = last['close'] + retr
			sl = df['low'].iloc[-2:].min() - self.fade_sl_extra_atr * atr1
			tp = self._session_vwap(df) or (last['close'] + 0.618 * impulse)
			return {'symbol': symbol, 'type': 'BUY', 'sl': sl, 'tp': tp, 'entry_price': entry}
