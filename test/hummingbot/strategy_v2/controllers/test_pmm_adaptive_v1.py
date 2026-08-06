import asyncio
from decimal import Decimal
from statistics import median
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pandas_ta as ta
import pydantic
from controllers.market_making.pmm_adaptive_v1 import PMMAdaptiveV1Config, PMMAdaptiveV1Controller

from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy_v2.executors.data_types import PositionSummary
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


def _build_candles_df(num_rows: int = 150, base_price: float = 100.0) -> pd.DataFrame:
    """A small, deterministic synthetic OHLC series with enough range/noise for NATR to be well-defined."""
    rows = []
    price = base_price
    for i in range(num_rows):
        # Deterministic oscillation so high/low/close all differ every bar (required for a non-zero NATR).
        drift = (i % 10 - 5) * 0.05
        price = price + drift
        high = price + 0.6
        low = price - 0.6
        close = price + 0.1
        rows.append({"timestamp": i, "open": price, "high": high, "low": low, "close": close, "volume": 10})
    return pd.DataFrame(rows)


def _order_book_df(levels) -> pd.DataFrame:
    return pd.DataFrame([{"price": p, "amount": a, "update_id": 1} for p, a in levels])


def _mock_executor(executor_id: str, level_id: str, is_active: bool = True, is_trading: bool = False) -> MagicMock:
    executor = MagicMock(spec=ExecutorInfo)
    executor.id = executor_id
    executor.is_active = is_active
    executor.is_trading = is_trading
    executor.custom_info = {"level_id": level_id}
    return executor


def _mock_position(connector_name: str, trading_pair: str, side: TradeType, amount: Decimal) -> MagicMock:
    position = MagicMock(spec=PositionSummary)
    position.connector_name = connector_name
    position.trading_pair = trading_pair
    position.side = side
    position.amount = amount
    return position


class TestPMMAdaptiveV1(IsolatedAsyncioWrapperTestCase):

    def setUp(self):
        self.config = PMMAdaptiveV1Config(
            id="test",
            connector_name="binance",
            trading_pair="BTC-USDT",
            total_amount_quote=Decimal("1000"),
            buy_spreads=[1, 2],
            sell_spreads=[1, 2],
            natr_length=14,
            inventory_target_base_pct=Decimal("0.5"),
            risk_aversion=Decimal("0.5"),
            max_skew_pct_of_natr=Decimal("0.5"),
            obi_depth_levels=5,
            obi_lookback_samples=10,
            obi_collapse_threshold_pct=Decimal("0.7"),
            obi_kill_switch_enabled=True,
        )
        self.mock_market_data_provider = MagicMock(spec=MarketDataProvider)
        self.mock_actions_queue = AsyncMock(spec=asyncio.Queue)
        self.controller = PMMAdaptiveV1Controller(
            config=self.config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        self.controller.positions_held = []
        self.controller.executors_info = []

    # --- Config defaulting / validation ---

    def test_candles_connector_defaults_to_connector_name(self):
        config = PMMAdaptiveV1Config(
            id="t", connector_name="binance", trading_pair="BTC-USDT", candles_connector="", candles_trading_pair="")
        self.assertEqual(config.candles_connector, "binance")
        self.assertEqual(config.candles_trading_pair, "BTC-USDT")

    def test_candles_connector_explicit_override_is_preserved(self):
        config = PMMAdaptiveV1Config(
            id="t", connector_name="binance", trading_pair="BTC-USDT",
            candles_connector="kucoin", candles_trading_pair="ETH-USDT")
        self.assertEqual(config.candles_connector, "kucoin")
        self.assertEqual(config.candles_trading_pair, "ETH-USDT")

    def test_config_decimal_fields_parse_from_string(self):
        config = PMMAdaptiveV1Config(
            id="t", connector_name="binance", trading_pair="BTC-USDT",
            inventory_target_base_pct="0.6", risk_aversion="1.2",
            max_skew_pct_of_natr="0.3", obi_collapse_threshold_pct="0.5")
        self.assertEqual(config.inventory_target_base_pct, Decimal("0.6"))
        self.assertEqual(config.risk_aversion, Decimal("1.2"))
        self.assertEqual(config.max_skew_pct_of_natr, Decimal("0.3"))
        self.assertEqual(config.obi_collapse_threshold_pct, Decimal("0.5"))

    def test_config_field_bounds_reject_invalid_values(self):
        base_kwargs = dict(id="t", connector_name="binance", trading_pair="BTC-USDT")
        with self.assertRaises(pydantic.ValidationError):
            PMMAdaptiveV1Config(**base_kwargs, inventory_target_base_pct=Decimal("1.5"))
        with self.assertRaises(pydantic.ValidationError):
            PMMAdaptiveV1Config(**base_kwargs, risk_aversion=Decimal("0"))
        with self.assertRaises(pydantic.ValidationError):
            PMMAdaptiveV1Config(**base_kwargs, obi_collapse_threshold_pct=Decimal("1.0"))
        with self.assertRaises(pydantic.ValidationError):
            PMMAdaptiveV1Config(**base_kwargs, obi_collapse_threshold_pct=Decimal("0"))

    # --- Candles config ---

    def test_get_candles_config_returns_expected_config(self):
        configs = self.controller.get_candles_config()
        self.assertEqual(len(configs), 1)
        candles_config = configs[0]
        self.assertIsInstance(candles_config, CandlesConfig)
        self.assertEqual(candles_config.connector, "binance")
        self.assertEqual(candles_config.trading_pair, "BTC-USDT")
        self.assertEqual(candles_config.max_records, self.config.natr_length + 100)

    # --- NATR / dynamic spread + neutral inventory ---

    async def test_natr_sets_spread_multiplier_and_neutral_inventory_has_no_shift(self):
        candles = _build_candles_df()
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=candles)
        self.mock_market_data_provider.get_order_book_snapshot = MagicMock(
            return_value=(_order_book_df([(99, 1)]), _order_book_df([(101, 1)])))
        self.controller.positions_held = []  # net position 0 -> flat vs target 0.5 (=neutral) -> q == 0

        await self.controller.update_processed_data()

        expected_natr = Decimal(str(
            (ta.natr(candles["high"], candles["low"], candles["close"], length=self.config.natr_length) / 100).iloc[-1]
        ))
        mid_price = Decimal(str(candles["close"].iloc[-1]))
        self.assertEqual(self.controller.processed_data["spread_multiplier"], expected_natr)
        self.assertEqual(self.controller.processed_data["reference_price"], mid_price)
        self.assertEqual(self.controller.processed_data["inventory_deviation"], Decimal("0"))

    # --- Inventory skew direction / clamping ---

    async def _run_with_position(self, base_amount: Decimal, side: TradeType):
        candles = _build_candles_df()
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=candles)
        self.mock_market_data_provider.get_order_book_snapshot = MagicMock(
            return_value=(_order_book_df([(99, 1)]), _order_book_df([(101, 1)])))
        self.controller.positions_held = [
            _mock_position(self.config.connector_name, self.config.trading_pair, side, base_amount)
        ]
        await self.controller.update_processed_data()
        return Decimal(str(candles["close"].iloc[-1]))

    async def test_inventory_skew_overweight_shifts_reference_price_down(self):
        mid_price = await self._run_with_position(Decimal("2"), TradeType.BUY)  # net long -> overweight base
        self.assertGreater(self.controller.processed_data["inventory_deviation"], Decimal("0"))
        self.assertLess(self.controller.processed_data["reference_price"], mid_price)

    async def test_inventory_skew_underweight_shifts_reference_price_up(self):
        mid_price = await self._run_with_position(Decimal("2"), TradeType.SELL)  # net short -> underweight base
        self.assertLess(self.controller.processed_data["inventory_deviation"], Decimal("0"))
        self.assertGreater(self.controller.processed_data["reference_price"], mid_price)

    async def test_inventory_skew_is_clamped_by_max_skew_pct_of_natr(self):
        self.config.risk_aversion = Decimal("1000")  # deliberately extreme, forces the clamp to bind
        mid_price = await self._run_with_position(Decimal("100"), TradeType.BUY)
        natr = self.controller.processed_data["spread_multiplier"]
        max_shift_pct = self.config.max_skew_pct_of_natr * natr
        actual_shift_pct = abs(self.controller.processed_data["reference_price"] - mid_price) / mid_price
        self.assertLessEqual(actual_shift_pct, max_shift_pct * Decimal("1.0001"))  # tiny tolerance for rounding

    def test_get_inventory_deviation_guards_zero_total_amount_quote(self):
        self.config.total_amount_quote = Decimal("0")
        self.controller.positions_held = [
            _mock_position(self.config.connector_name, self.config.trading_pair, TradeType.BUY, Decimal("5"))
        ]
        q = self.controller._get_inventory_deviation(Decimal("100"))
        self.assertEqual(q, Decimal("0"))

    # --- Depth snapshot resilience (e.g. candle-driven backtesting has no live order book) ---

    async def test_update_processed_data_survives_missing_order_book(self):
        # Mirrors BacktestingDataProvider: no live order book subscription exists for the pair, so
        # get_order_book_snapshot raises instead of returning data. NATR/inventory-skew must still complete.
        candles = _build_candles_df()
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=candles)
        self.mock_market_data_provider.get_order_book_snapshot = MagicMock(
            side_effect=ValueError("No order book exists for 'BTC-USDT'."))

        await self.controller.update_processed_data()

        self.assertEqual(self.controller.processed_data["bid_depth"], Decimal("0"))
        self.assertEqual(self.controller.processed_data["ask_depth"], Decimal("0"))
        self.assertIn("reference_price", self.controller.processed_data)
        self.assertIn("spread_multiplier", self.controller.processed_data)

    def test_get_depth_snapshot_returns_zero_on_missing_order_book(self):
        self.mock_market_data_provider.get_order_book_snapshot = MagicMock(
            side_effect=ValueError("No order book exists for 'BTC-USDT'."))
        bid_depth, ask_depth = self.controller._get_depth_snapshot()
        self.assertEqual(bid_depth, Decimal("0"))
        self.assertEqual(ask_depth, Decimal("0"))

    # --- OBI fill-kill ---

    def _seed_obi_history(self, baseline_bid: Decimal, baseline_ask: Decimal, current_bid: Decimal, current_ask: Decimal,
                          samples: int = 6):
        for _ in range(samples - 1):
            self.controller._bid_depth_history.append(baseline_bid)
            self.controller._ask_depth_history.append(baseline_ask)
        self.controller._bid_depth_history.append(current_bid)
        self.controller._ask_depth_history.append(current_ask)
        bid_ratio, ask_ratio = self.controller._compute_collapse_ratios()
        self.controller.processed_data = {"bid_collapse_ratio": bid_ratio, "ask_collapse_ratio": ask_ratio}

    def test_obi_collapse_on_bid_side_stops_only_buy_non_trading_executors(self):
        self._seed_obi_history(Decimal("10"), Decimal("10"), Decimal("1"), Decimal("10"))  # bid depth -90%
        self.assertGreaterEqual(self.controller.processed_data["bid_collapse_ratio"], self.config.obi_collapse_threshold_pct)

        buy_resting = _mock_executor("buy_resting", "buy_0", is_active=True, is_trading=False)
        sell_resting = _mock_executor("sell_resting", "sell_0", is_active=True, is_trading=False)
        buy_trading = _mock_executor("buy_trading", "buy_1", is_active=True, is_trading=True)
        self.controller.executors_info = [buy_resting, sell_resting, buy_trading]

        actions = self.controller.executors_to_early_stop()

        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopExecutorAction)
        self.assertEqual(actions[0].executor_id, "buy_resting")

    def test_obi_collapse_on_ask_side_stops_only_sell_non_trading_executors(self):
        self._seed_obi_history(Decimal("10"), Decimal("10"), Decimal("10"), Decimal("1"))  # ask depth -90%

        buy_resting = _mock_executor("buy_resting", "buy_0", is_active=True, is_trading=False)
        sell_resting = _mock_executor("sell_resting", "sell_0", is_active=True, is_trading=False)
        sell_trading = _mock_executor("sell_trading", "sell_1", is_active=True, is_trading=True)
        self.controller.executors_info = [buy_resting, sell_resting, sell_trading]

        actions = self.controller.executors_to_early_stop()

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].executor_id, "sell_resting")

    def test_obi_no_collapse_returns_no_stop_actions(self):
        self._seed_obi_history(Decimal("10"), Decimal("10"), Decimal("9.5"), Decimal("9.5"))  # ~5% drop, below threshold
        self.controller.executors_info = [_mock_executor("buy_resting", "buy_0")]

        actions = self.controller.executors_to_early_stop()
        self.assertEqual(actions, [])

    def test_obi_kill_switch_disabled_suppresses_all_stops(self):
        self.config.obi_kill_switch_enabled = False
        self._seed_obi_history(Decimal("10"), Decimal("10"), Decimal("1"), Decimal("10"))
        self.controller.executors_info = [_mock_executor("buy_resting", "buy_0")]

        actions = self.controller.executors_to_early_stop()
        self.assertEqual(actions, [])

    def test_obi_insufficient_history_returns_no_stop_actions(self):
        # Only 2 samples appended, below MIN_OBI_SAMPLES -> no signal even though the drop looks huge.
        self.controller._bid_depth_history.append(Decimal("10"))
        self.controller._bid_depth_history.append(Decimal("0.1"))
        self.controller._ask_depth_history.append(Decimal("10"))
        self.controller._ask_depth_history.append(Decimal("10"))
        bid_ratio, ask_ratio = self.controller._compute_collapse_ratios()
        self.assertEqual(bid_ratio, Decimal("0"))
        self.assertEqual(ask_ratio, Decimal("0"))

    def test_obi_baseline_uses_median_not_mean(self):
        # Baseline samples: mostly 10, one outlier spike of 100 -> median stays 10, mean would be dragged up.
        history = [Decimal("10")] * 5 + [Decimal("100")]
        for v in history:
            self.controller._bid_depth_history.append(v)
            self.controller._ask_depth_history.append(v)
        self.controller._bid_depth_history.append(Decimal("9"))  # current reading, ~10% drop vs median baseline of 10
        self.controller._ask_depth_history.append(Decimal("9"))

        bid_ratio, _ = self.controller._compute_collapse_ratios()
        expected_baseline = median(history)
        expected_ratio = (expected_baseline - Decimal("9")) / expected_baseline
        self.assertEqual(bid_ratio, expected_ratio)

    def test_obi_processed_data_missing_returns_no_stop_actions(self):
        # executors_to_early_stop called before the first update_processed_data tick.
        self.controller.processed_data = {}
        self.controller.executors_info = [_mock_executor("buy_resting", "buy_0")]
        actions = self.controller.executors_to_early_stop()
        self.assertEqual(actions, [])

    # --- Executor config ---

    def test_get_executor_config_returns_position_executor_config(self):
        self.mock_market_data_provider.time = MagicMock(return_value=1234567890)
        result = self.controller.get_executor_config("buy_0", Decimal("100"), Decimal("0.01"))
        self.assertIsInstance(result, PositionExecutorConfig)
        self.assertEqual(result.connector_name, "binance")
        self.assertEqual(result.trading_pair, "BTC-USDT")
        self.assertEqual(result.side, TradeType.BUY)
        self.assertEqual(result.entry_price, Decimal("100"))
        self.assertEqual(result.amount, Decimal("0.01"))
        self.assertEqual(result.leverage, self.config.leverage)
        self.assertEqual(result.triple_barrier_config, self.config.triple_barrier_config)
