from strategy_base import StrategyBase
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np

# Import MT5 timeframes (optional)
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


class BreakoutScalper(StrategyBase):
    """Breakout Scalping Strategy"""

    def __init__(self, name: str, symbols: List[str], params: Dict[str, Any]):
        super().__init__(name, symbols, params)
        
        # Strategy parameters
        self.candle_count_min = params.get('candle_count_min', 20)
        self.candle_count_max = params.get('candle_count_max', 40)
        self.atr_compression_threshold = params.get('atr_compression_threshold', 0.4) # 40% of 20-period ATR
        self.range_size_min_factor = params.get('range_size_min_factor', 0.3) # 0.3 * ATR
        self.range_size_max_factor = params.get('range_size_max_factor', 0.7) # 0.7 * ATR
        self.volatility_collapse_threshold = params.get('volatility_collapse_threshold', 0.5) # 50% drop
        self.breakout_candle_body_multiplier = params.get('breakout_candle_body_multiplier', 1.5)
        self.breakout_volume_multiplier = params.get('breakout_volume_multiplier', 1.5)
        self.sl_atr_multiplier = params.get('sl_atr_multiplier', 0.3)
        self.min_rr = params.get('min_rr', 1.5)
        self.fail_safe_candles = params.get('fail_safe_candles', 3)
        self.timeframe = params.get('timeframe', (mt5.TIMEFRAME_M1 if mt5 else 0)) # Default to 1-minute (0 if MT5 absent)
        self.confirmation_timeframe = params.get('confirmation_timeframe', (mt5.TIMEFRAME_M15 if mt5 else 0)) # Default 15m

        self.consolidation_zone = {symbol: None for symbol in symbols} # Store detected consolidation zones
        self.last_breakout_candle = {symbol: None for symbol in symbols}
        self.max_history_bars = max(self.candle_count_max * 2, 60) # Ensure enough history for indicators

        logging.info(f"Breakout Scalper Strategy {name} initialized for symbols {symbols}")

    def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on market data for breakout scalping.
        """
        for symbol in self.symbols:
            if symbol not in data or self._history[symbol].empty:
                continue

            # Get the latest completed bar
            latest_bar = self._history[symbol].iloc[-1]
            current_price = (data[symbol]['bid'] + data[symbol]['ask']) / 2

            # Step 1: Consolidation Detection
            if self.consolidation_zone[symbol] is None:
                zone = self._detect_consolidation(symbol, latest_bar)
                if zone:
                    self.consolidation_zone[symbol] = zone
                    logging.info(f"Consolidation detected for {symbol}: {zone}")
                    continue # Wait for breakout
            
            # If consolidation zone is detected, look for breakout
            if self.consolidation_zone[symbol]:
                signal = self._check_for_breakout(symbol, latest_bar, current_price)
                if signal:
                    # Clear consolidation zone after breakout signal
                    self.consolidation_zone[symbol] = None
                    return signal
                else:
                    # Check for fail-safe exit (re-entry into range)
                    if self._check_fail_safe_exit(symbol, latest_bar):
                        logging.info(f"Fail-safe exit triggered for {symbol}: price re-entered range.")
                        self.consolidation_zone[symbol] = None # Reset zone
                        # If we had an open position, it would be closed by engine

            # If there's an open position, monitor it for trailing SL
            if self.position == symbol and self.position_data.get(symbol):
                # This part is handled by the ProfitLockManager in trading_engine
                pass

        return None

    def _detect_consolidation(self, symbol: str, latest_bar: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Identifies a valid consolidation zone based on defined criteria.
        """
        df = self._history[symbol]
        if len(df) < self.candle_count_max:
            return None

        # Analyze the last `self.candle_count_max` bars for consolidation
        recent_df = df.iloc[-self.candle_count_max:]

        # Candle Count
        if not (self.candle_count_min <= len(recent_df) <= self.candle_count_max):
            return None

        # ATR Compression
        atr_current = self._calculate_atr(symbol, 14) # Use 14-period ATR for current
        atr_20_period_avg = self._calculate_atr(symbol, 20) # Use 20-period ATR for average
        if atr_current is None or atr_20_period_avg is None or atr_current > self.atr_compression_threshold * atr_20_period_avg:
            return None

        # Range Size
        highs = recent_df['high']
        lows = recent_df['low']
        range_high = highs.max()
        range_low = lows.min()
        range_size = range_high - range_low

        if atr_current is None or not (self.range_size_min_factor * atr_current <= range_size <= self.range_size_max_factor * atr_current):
            return None
        
        # Volatility Collapse (using Bollinger Band width or Standard Deviation)
        bollinger_bands = self._calculate_bollinger_bands(symbol, 20, 2) # 20-period, 2 std dev
        std_dev_current = self._calculate_std_dev(symbol, 20)
        std_dev_20_period_avg = df['close'].iloc[-40:-20].std() # Average of previous 20 bars before recent

        if bollinger_bands:
            # For simplicity, let's use std dev for now
            if std_dev_current is None or std_dev_20_period_avg is None or std_dev_current > (1 - self.volatility_collapse_threshold) * std_dev_20_period_avg:
                return None
        else:
             return None # Bollinger Bands not calculable

        # Structure Validation (simple approximation: check if price stayed within range)
        # This is hard to automate perfectly, a basic check is to see if min/max are within a tight range
        # For now, let's consider the tightness of the range as sufficient validation

        # Context (align with 15-min trend) - requires 15-min data
        # This will be handled in the main engine by fetching 15-min data and passing to strategy
        # For now, we assume this check is done externally or will be integrated later.

        # Bonus Filter: Volume decline (simple check: current avg volume vs earlier avg volume)
        avg_volume_recent = self._calculate_average_volume(symbol, len(recent_df))
        avg_volume_prior = self._calculate_average_volume(symbol, 20) # Compare to earlier 20 bars
        if avg_volume_recent is None or avg_volume_prior is None or avg_volume_recent > avg_volume_prior * 0.9: # 10% decline
             # If volume hasn't declined, it's not a strong consolidation for this filter
             pass # Not a hard filter for now

        # If all conditions pass, return the consolidation zone details
        return {
            'resistance': range_high,
            'support': range_low,
            'range_size': range_size,
            'timeframe_start': recent_df.index[0],
            'timeframe_end': recent_df.index[-1]
        }

    def _check_for_breakout(self, symbol: str, latest_bar: pd.Series, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Checks for valid breakout from the consolidation zone.
        """
        zone = self.consolidation_zone[symbol]
        if not zone:
            return None

        resistance = zone['resistance']
        support = zone['support']
        range_size = zone['range_size']

        signal_type = None
        breakout_price = None
        
        # Long Entry: Price breaks and closes above resistance
        if latest_bar['close'] > resistance:
            signal_type = 'BUY'
            breakout_price = resistance
        # Short Entry: Price breaks and closes below support
        elif latest_bar['close'] < support:
            signal_type = 'SELL'
            breakout_price = support

        if signal_type:
            # Breakout candle body size
            body_size = abs(latest_bar['open'] - latest_bar['close'])
            avg_body_size = self._calculate_average_body_size(symbol, 10)
            if avg_body_size is None or body_size < self.breakout_candle_body_multiplier * avg_body_size:
                logging.debug(f"Breakout for {symbol} - candle body too small")
                return None

            # Breakout volume
            breakout_volume = latest_bar['tick_volume']
            avg_volume_20 = self._calculate_average_volume(symbol, 20)
            if avg_volume_20 is None or breakout_volume < self.breakout_volume_multiplier * avg_volume_20:
                logging.debug(f"Breakout for {symbol} - volume too low")
                return None
            
            # Stop-Loss & Take-Profit
            atr_current = self._calculate_atr(symbol, 14) # Use 14-period ATR for SL calculation
            if atr_current is None:
                logging.warning(f"Cannot calculate ATR for {symbol}, skipping trade")
                return None

            sl_distance = self.sl_atr_multiplier * atr_current
            if sl_distance <= 0: # Ensure valid SL distance
                sl_distance = self.min_profit_lock_pips * 0.0001 * 2 # Fallback to 10 pips

            if signal_type == 'BUY':
                sl = breakout_price - sl_distance
                # TP = breakout price + range size
                tp = breakout_price + range_size
            else: # SELL
                sl = breakout_price + sl_distance
                # TP = breakout price - range size
                tp = breakout_price - range_size

            # Ensure minimum RR
            actual_rr = abs(tp - current_price) / abs(sl - current_price)
            if actual_rr < self.min_rr:
                logging.debug(f"Breakout for {symbol} - RR {actual_rr:.1f} < min {self.min_rr}, adjusting TP")
                # Adjust TP to meet minimum RR
                if signal_type == 'BUY':
                    tp = current_price + (abs(current_price - sl) * self.min_rr)
                else:
                    tp = current_price - (abs(current_price - sl) * self.min_rr)

            logging.info(f"Breakout signal for {symbol}: {signal_type} at {current_price:.5f}, SL={sl:.5f}, TP={tp:.5f}")
            self.last_breakout_candle[symbol] = latest_bar # Store breakout candle for fail-safe
            
            return {
                'symbol': symbol,
                'type': signal_type,
                'sl': sl,
                'tp': tp,
                'entry_price': current_price # Retest entry is optional, use current price for now
            }
        
        return None

    def _check_fail_safe_exit(self, symbol: str, latest_bar: pd.Series) -> bool:
        """
        Checks if momentum failed and price re-entered the consolidation range.
        """
        zone = self.consolidation_zone[symbol]
        if not zone or not self.last_breakout_candle[symbol]:
            return False
        
        # Check if latest_bar is within the fail-safe window
        breakout_time = self.last_breakout_candle[symbol].name # Index is time
        if latest_bar.name - breakout_time > pd.Timedelta(minutes=self.fail_safe_candles * 1): # Assuming 1-min bars
            return False # Beyond fail-safe window

        # Check if price re-entered the range
        if zone['support'] < latest_bar['close'] < zone['resistance']:
            return True
        
        return False

    def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Called on each tick for signal generation.
        For breakout strategy, we primarily rely on completed bars, so this method
        will just pass data to generate_signal.
        """
        try:
            # The TradingEngine will call _update_history with new bars
            # and then pass the latest tick data to generate_signal.
            return self.generate_signal(tick_data)
        except Exception as e:
            logging.error(f"Error in strategy {self.name}: {e}")
            return None
