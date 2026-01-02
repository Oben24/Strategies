from strategy_base import StrategyBase
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class RangeReversionStrategy(StrategyBase):
    """Range mean-reversion using Bollinger Bands and RSI(2) with ADX filter."""

    def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
        super().__init__(name, symbols, params)

        self.bb_period = int(params.get('bb_period', 20))
        self.bb_mult = float(params.get('bb_mult', 2.0))
        self.rsi_len = int(params.get('rsi_len', 2))
        self.rsi_buy = float(params.get('rsi_buy', 5))
        self.rsi_sell = float(params.get('rsi_sell', 95))
        self.use_adx = bool(params.get('use_adx', True))
        self.adx_period = int(params.get('adx_period', 14))
        self.adx_max = float(params.get('adx_max', 20))
        self.use_bbwidth_filter = bool(params.get('use_bbwidth_filter', True))
        self.bbwidth_max = float(params.get('bbwidth_max', 0.02))
        self.lookback_touch = int(params.get('lookback_touch', 2))
        self.atr_period = int(params.get('atr_period', 14))
        self.atr_mult_sl = float(params.get('atr_mult_sl', 1.0))
        self.atr_mult_tp = float(params.get('atr_mult_tp', 1.6))
        self.atr_mult_trail = float(params.get('atr_mult_trail', 0.8))
        self.trail_start_rr = float(params.get('trail_start_rr', 1.0))
        self.exit_to_mid = bool(params.get('exit_to_mid', True))
        self.partial_to_mid = bool(params.get('partial_to_mid', False))
        self.max_bars_open = int(params.get('max_bars_open', 24))
        self.max_positions_per_symbol = int(params.get('max_positions_per_symbol', 1))
        self.trading_window = params.get('trading_window_utc', ["06:00", "21:00"])  # [start, end]
        self.friday_cutoff = params.get('friday_cutoff_utc', '17:00')

        self.timeframe = 30
        self.max_history_bars = max(self.bb_period + 10, 100)

        # per-symbol state
        self._bars_open: Dict[str, int] = {sym: 0 for sym in symbols}

    @staticmethod
    def _bb(close: pd.Series, period: int, mult: float):
        ma = close.rolling(period).mean()
        sd = close.rolling(period).std(ddof=0)
        up = ma + mult * sd
        dn = ma - mult * sd
        with np.errstate(divide='ignore', invalid='ignore'):
            bbw = ((up - dn) / ma).abs()
        return ma, up, dn, bbw.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _rsi(series: pd.Series, length: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(length).mean()
        roll_down = down.rolling(length).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _touched_recent(self, df: pd.DataFrame, bb_up: pd.Series, bb_dn: pd.Series, lookback: int) -> bool:
        if lookback <= 0 or len(df) < lookback + 2:
            return True
        hi = df['high'].iloc[-lookback:]
        lo = df['low'].iloc[-lookback:]
        up = bb_up.iloc[-lookback:]
        dn = bb_dn.iloc[-lookback:]
        return bool(((hi >= up) | (lo <= dn)).any())

    def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for symbol in self.symbols:
            if symbol not in data or self._history[symbol].empty:
                continue

            df = self._history[symbol]
            if len(df) < self.max_history_bars:
                continue

            ts = df.index[-1]
            # trading window
            try:
                start_s, end_s = self.trading_window
                h1, m1 = map(int, start_s.split(':'))
                h2, m2 = map(int, end_s.split(':'))
                t = ts.time()
                if not ((t.hour > h1 or (t.hour == h1 and t.minute >= m1)) and (t.hour < h2 or (t.hour == h2 and t.minute <= m2))):
                    continue
            except Exception:
                pass

            # in-position tracking only (engine manages exits)
            if self.position == symbol:
                self._bars_open[symbol] = self._bars_open.get(symbol, 0) + 1
                continue

            close = df['close'].astype(float)
            ma, bb_up, bb_dn, bbw = self._bb(close, self.bb_period, self.bb_mult)
            rsi2 = self._rsi(close, self.rsi_len)

            # ATR
            tr1 = (df['high'] - df['low']).abs()
            tr2 = (df['high'] - df['close'].shift()).abs()
            tr3 = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_s = tr.rolling(self.atr_period).mean()
            if not (np.isfinite(atr_s.iloc[-1]) and np.isfinite(ma.iloc[-1])):
                continue

            price = float(close.iloc[-1])
            high = float(df['high'].iloc[-1])
            low = float(df['low'].iloc[-1])
            atr = float(atr_s.iloc[-1])
            mid = float(ma.iloc[-1])
            up = float(bb_up.iloc[-1]) if np.isfinite(bb_up.iloc[-1]) else None
            dn = float(bb_dn.iloc[-1]) if np.isfinite(bb_dn.iloc[-1]) else None

            # Range regime filters
            if self.use_adx:
                adx = self._calculate_adx(symbol, self.adx_period)
                if adx is None or float(adx['adx']) > self.adx_max:
                    continue
            if self.use_bbwidth_filter:
                if not np.isfinite(bbw.iloc[-1]) or float(bbw.iloc[-1]) > self.bbwidth_max:
                    continue

            if not self._touched_recent(df, bb_up, bb_dn, self.lookback_touch):
                continue

            rsi_now = float(rsi2.iloc[-1]) if np.isfinite(rsi2.iloc[-1]) else None
            if rsi_now is None or up is None or dn is None:
                continue

            # Long fade
            if price <= dn and rsi_now <= self.rsi_buy:
                sl = min(low, price - self.atr_mult_sl * atr)
                tp = mid if self.exit_to_mid else price + self.atr_mult_tp * atr
                self._bars_open[symbol] = 0
                return {'symbol': symbol, 'type': 'BUY', 'sl': float(sl), 'tp': float(tp)}

            # Short fade
            if price >= up and rsi_now >= self.rsi_sell:
                sl = max(high, price + self.atr_mult_sl * atr)
                tp = mid if self.exit_to_mid else price - self.atr_mult_tp * atr
                self._bars_open[symbol] = 0
                return {'symbol': symbol, 'type': 'SELL', 'sl': float(sl), 'tp': float(tp)}

        return None

    def should_close_position(self, tick_data: Dict[str, Any]) -> bool:
        if super().should_close_position(tick_data):
            return True
        sym = self.position
        if not sym:
            return False
        df = self._history.get(sym)
        if df is None or df.empty:
            return False
        # Time stop
        if self._bars_open.get(sym, 0) >= self.max_bars_open:
            return True
        # Midline exit if configured and TP not set to mid
        close = df['close'].astype(float)
        ma = close.rolling(self.bb_period).mean()
        if np.isfinite(ma.iloc[-1]):
            mid = float(ma.iloc[-1])
            side = self.position_data.get(sym, {}).get('type')
            price = float(close.iloc[-1])
            if self.exit_to_mid and ((side == 'BUY' and price >= mid) or (side == 'SELL' and price <= mid)):
                return True
        return False


