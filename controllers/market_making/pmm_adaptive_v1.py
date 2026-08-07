from collections import deque
from decimal import Decimal
from statistics import median
from typing import Deque, List, Tuple, Union

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction

# Floor applied to NATR so that `spread_multiplier` is never literally 0 (which would make
# get_price_and_amount() place buy and sell orders at the same reference price).
MIN_NATR = Decimal("0.0001")
# Minimum number of OBI depth samples required before the collapse detector is allowed to fire,
# so the very first ticks (before a real rolling baseline exists) never trigger a false kill.
MIN_OBI_SAMPLES = 5


class PMMAdaptiveV1Config(MarketMakingControllerConfigBase):
    """
    Configuration for the PMM Adaptive V1 controller.
    """
    controller_name: str = "pmm_adaptive_v1"
    connector_name: str = Field(
        default="binance",
        json_schema_extra={
            "prompt": "Enter the connector name (e.g., binance): ",
            "prompt_on_new": True}
    )
    trading_pair: str = Field(
        default="BTC-USDT",
        json_schema_extra={
            "prompt": "Enter the trading pair to trade on (e.g., BTC-USDT): ",
            "prompt_on_new": True}
    )
    buy_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of buy spreads measured in units of volatility (e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sell_spreads: List[float] = Field(
        default="1,2,4",
        json_schema_extra={
            "prompt": "Enter a comma-separated list of sell spreads measured in units of volatility (e.g., '1, 2'): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    # Re-declared (not just inherited) purely to add validate_default=True: the base class's own
    # field_validator(mode="before") that fills these in as equal-weight lists across the configured spread
    # levels only runs when a value is explicitly passed (even blank), not when the field is omitted
    # entirely -- which is exactly what a config built programmatically (e.g. a backtest script's config
    # dict) commonly does. Without this, get_spreads_and_amounts_in_quote() crashes on `sum(None)`.
    buy_amounts_pct: Union[List[Decimal], None] = Field(
        default=None,
        validate_default=True,
        json_schema_extra={
            "prompt": "Enter a comma-separated list of buy amounts as percentages (e.g., '50, 50'), or leave blank to distribute equally: ",
            "prompt_on_new": True, "is_updatable": True}
    )
    sell_amounts_pct: Union[List[Decimal], None] = Field(
        default=None,
        validate_default=True,
        json_schema_extra={
            "prompt": "Enter a comma-separated list of sell amounts as percentages (e.g., '50, 50'), or leave blank to distribute equally: ",
            "prompt_on_new": True, "is_updatable": True}
    )

    # === Dynamic volatility spread (NATR) ===
    candles_connector: str = Field(
        default=None,
        validate_default=True,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True})
    candles_trading_pair: str = Field(
        default=None,
        validate_default=True,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True})
    interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    natr_length: int = Field(
        default=14, gt=0,
        json_schema_extra={"prompt": "Enter the NATR length: ", "prompt_on_new": True})

    # === Avellaneda-Stoikov inventory skew ===
    inventory_target_base_pct: Decimal = Field(
        default=Decimal("0.5"), ge=0, le=1,
        json_schema_extra={
            "prompt": "Enter the target base asset inventory as a fraction of your capital "
                      "(0.5 = flat/neutral, 1 = fully long-biased, 0 = fully short-biased): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    risk_aversion: Decimal = Field(
        default=Decimal("0.5"), gt=0,
        json_schema_extra={
            "prompt": "Enter the inventory risk aversion (Avellaneda-Stoikov gamma, e.g. 0.5): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    max_skew_pct_of_natr: Decimal = Field(
        default=Decimal("0.5"), gt=0, le=1,
        json_schema_extra={
            "prompt": "Enter the max inventory price-shift as a fraction of NATR (clamp, e.g. 0.5): ",
            "prompt_on_new": True, "is_updatable": True}
    )

    # === Order book imbalance (OBI) fill-kill ===
    obi_depth_levels: int = Field(
        default=10, gt=0,
        json_schema_extra={
            "prompt": "Enter the number of order book levels to sum for depth (e.g. 10): ",
            "prompt_on_new": True}
    )
    obi_lookback_samples: int = Field(
        default=30, gt=1,
        json_schema_extra={
            "prompt": "Enter the number of rolling samples to keep for the OBI depth baseline (e.g. 30): ",
            "prompt_on_new": True}
    )
    obi_collapse_threshold_pct: Decimal = Field(
        default=Decimal("0.7"), gt=0, lt=1,
        json_schema_extra={
            "prompt": "Enter the depth-collapse threshold that triggers a fill-kill (e.g. 0.7 for 70%): ",
            "prompt_on_new": True, "is_updatable": True}
    )
    obi_kill_switch_enabled: bool = Field(
        default=True,
        json_schema_extra={
            "prompt": "Enable the order book imbalance fill-kill switch? (True/False): ",
            "prompt_on_new": True, "is_updatable": True}
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v

    @field_validator(
        "inventory_target_base_pct", "risk_aversion", "max_skew_pct_of_natr", "obi_collapse_threshold_pct",
        mode="before")
    @classmethod
    def parse_decimal_fields(cls, v):
        if isinstance(v, str) and v != "":
            return Decimal(v)
        return v


class PMMAdaptiveV1Controller(MarketMakingControllerBase):
    """
    Adaptive PMM controller that extends the base market making controller with three risk-management
    improvements over a static Pure Market Making template:

    1. Dynamic volatility spreads: `spread_multiplier` is set from NATR, so the configured `buy_spreads`/
       `sell_spreads` (entered in units of volatility) automatically widen in choppy markets and tighten in
       quiet ones, instead of staying static.
    2. Avellaneda-Stoikov-style inventory skew: `reference_price` is shifted away from the mid-price based on
       how far the controller's own position (not raw wallet balance) has drifted from
       `inventory_target_base_pct`. See `_get_inventory_deviation` for the sign convention -- this is the
       mechanism that stops the bot from "buying all the way down" in a trend.
    3. Order book imbalance (OBI) fill-kill: `executors_to_early_stop` cancels resting, not-yet-filled quotes
       on a side when that side's order book depth collapses sharply (e.g. >70%) versus its recent rolling
       baseline, so quotes aren't left sitting in front of a book that's about to be swept by toxic flow.
       Already-filled (is_trading) executors are left alone to run out their Triple Barrier.
    """

    def __init__(self, config: PMMAdaptiveV1Config, *args, **kwargs):
        self.config = config
        self.max_records = config.natr_length + 100
        self._bid_depth_history: Deque[Decimal] = deque(maxlen=config.obi_lookback_samples)
        self._ask_depth_history: Deque[Decimal] = deque(maxlen=config.obi_lookback_samples)
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        candles = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records)

        natr = self._get_natr(candles)
        mid_price = Decimal(str(candles["close"].iloc[-1]))

        inventory_deviation = self._get_inventory_deviation(mid_price)
        skew_pct = self._get_clamped_skew_pct(inventory_deviation, natr)
        reference_price = mid_price * (1 + skew_pct)

        bid_depth, ask_depth = self._get_depth_snapshot()
        self._bid_depth_history.append(bid_depth)
        self._ask_depth_history.append(ask_depth)
        bid_collapse_ratio, ask_collapse_ratio = self._compute_collapse_ratios()

        self.processed_data = {
            "reference_price": reference_price,
            "spread_multiplier": natr,
            "features": candles,
            "inventory_deviation": inventory_deviation,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bid_collapse_ratio": bid_collapse_ratio,
            "ask_collapse_ratio": ask_collapse_ratio,
        }

    def _get_natr(self, candles) -> Decimal:
        natr_series = ta.natr(candles["high"], candles["low"], candles["close"], length=self.config.natr_length) / 100
        natr_raw = natr_series.iloc[-1] if len(natr_series) > 0 else float("nan")
        if natr_raw != natr_raw:  # NaN check (warm-up period, not enough candle history yet)
            return MIN_NATR
        return max(Decimal(str(natr_raw)), MIN_NATR)

    def _get_inventory_deviation(self, mid_price: Decimal) -> Decimal:
        """
        Returns `q`, the controller's own position deviation from its target, normalized to roughly [-1, 1].

        `q` is based on this controller's own net filled amount from its own executors
        (`get_current_base_position`, inherited from MarketMakingControllerBase) -- not raw wallet balance.
        This is intentional: raw wallet balance reads ~0 on perpetual connectors (the position isn't held as
        a spot balance) and can be polluted by unrelated holdings in the same wallet on spot connectors.
        Using the controller's own position works uniformly for spot and perpetual, and only reflects
        inventory this controller itself is responsible for managing.

        Sign convention (matches the reservation-price shift in the legacy Avellaneda-Stoikov strategy at
        hummingbot/strategy/avellaneda_market_making/avellaneda_market_making.pyx):
        `q > 0` means the controller is overweight base asset relative to its target.
        """
        if mid_price <= 0 or self.config.total_amount_quote <= 0:
            return Decimal("0")
        current_base_position = self.get_current_base_position()
        current_position_value_quote = current_base_position * mid_price
        # inventory_target_base_pct=0.5 -> target 0 (flat); 1.0 -> target = +total_amount_quote (fully long);
        # 0.0 -> target = -total_amount_quote (fully short; only reachable on a perpetual connector).
        target_position_value_quote = self.config.total_amount_quote * (self.config.inventory_target_base_pct * 2 - 1)
        q = (current_position_value_quote - target_position_value_quote) / self.config.total_amount_quote
        return max(min(q, Decimal("1")), Decimal("-1"))

    def _get_clamped_skew_pct(self, inventory_deviation: Decimal, natr: Decimal) -> Decimal:
        """
        Reservation-price shift, as a fraction of mid-price.

        Overweight (q > 0) shifts the reference price down: sell orders (quoted above reference_price) get
        cheaper -- more eager to reduce the overweight position -- and buy orders (quoted below
        reference_price) get less competitive -- curbing further accumulation while already overweight. This
        directly targets the "buys all the way down" failure mode of a naive market maker. Underweight
        (q < 0) mirrors this in the opposite direction. The shift is clamped to `max_skew_pct_of_natr * natr`
        so extreme gamma or extreme inventory deviation can't push quotes into an unreasonable range or
        invert their ordering.
        """
        raw_skew_pct = -inventory_deviation * self.config.risk_aversion * natr
        max_shift = self.config.max_skew_pct_of_natr * natr
        return max(min(raw_skew_pct, max_shift), -max_shift)

    def _get_depth_snapshot(self) -> Tuple[Decimal, Decimal]:
        """
        Returns (0, 0) whenever a live order book snapshot isn't available for this connector/pair -- notably
        during candle-driven backtesting (BacktestingDataProvider never subscribes to a real order book), but
        also possible live/paper if a pair's book hasn't finished its initial WS subscription yet. Depth
        history staying at 0 means `_compute_collapse_ratios` never sees a real baseline (its `> 0` guards
        keep the ratio at 0), so the OBI fill-kill simply never fires rather than crashing the control loop --
        it's a best-effort risk overlay on top of the NATR/inventory-skew logic, not something that should be
        able to take the whole controller down.
        """
        try:
            bids_df, asks_df = self.market_data_provider.get_order_book_snapshot(
                self.config.connector_name, self.config.trading_pair)
        except Exception:
            return Decimal("0"), Decimal("0")
        n = self.config.obi_depth_levels
        bid_depth = Decimal(str(bids_df["amount"].iloc[:n].sum())) if not bids_df.empty else Decimal("0")
        ask_depth = Decimal(str(asks_df["amount"].iloc[:n].sum())) if not asks_df.empty else Decimal("0")
        return bid_depth, ask_depth

    def _compute_collapse_ratios(self) -> Tuple[Decimal, Decimal]:
        """
        Returns how far the current bid/ask depth has dropped versus a rolling median baseline, as a
        fraction in [0, 1] (0 = no drop, 1 = fully depleted). Uses the median (not the mean) of the prior
        samples so a single noisy/spiky reading doesn't distort the baseline. Returns (0, 0) -- i.e. no
        signal -- until enough samples have been collected to form a meaningful baseline.
        """
        min_samples = min(MIN_OBI_SAMPLES, self.config.obi_lookback_samples)
        if len(self._bid_depth_history) < min_samples:
            return Decimal("0"), Decimal("0")

        bid_hist, ask_hist = list(self._bid_depth_history), list(self._ask_depth_history)
        # Baseline excludes the just-appended current reading so we compare "now" against "before now".
        bid_baseline = median(bid_hist[:-1]) if len(bid_hist) > 1 else bid_hist[0]
        ask_baseline = median(ask_hist[:-1]) if len(ask_hist) > 1 else ask_hist[0]

        bid_ratio = (bid_baseline - bid_hist[-1]) / bid_baseline if bid_baseline > 0 else Decimal("0")
        ask_ratio = (ask_baseline - ask_hist[-1]) / ask_baseline if ask_baseline > 0 else Decimal("0")
        return max(bid_ratio, Decimal("0")), max(ask_ratio, Decimal("0"))

    def executors_to_early_stop(self) -> List[ExecutorAction]:
        if not self.config.obi_kill_switch_enabled or "bid_collapse_ratio" not in self.processed_data:
            return []

        stop_actions: List[ExecutorAction] = []
        if self.processed_data["bid_collapse_ratio"] >= self.config.obi_collapse_threshold_pct:
            # Bid depth (the side our BUY orders rest on) has thinned sharply -> a drop through our bid is
            # likely -> pull resting, unfilled buy orders before they eat toxic sell flow.
            stop_actions.extend(self._stop_side(TradeType.BUY))
        if self.processed_data["ask_collapse_ratio"] >= self.config.obi_collapse_threshold_pct:
            stop_actions.extend(self._stop_side(TradeType.SELL))
        return stop_actions

    def _stop_side(self, trade_type: TradeType) -> List[StopExecutorAction]:
        # Only pull executors that are still resting/unfilled -- an executor that's already is_trading
        # (partially or fully filled) keeps running its Triple Barrier rather than being abandoned.
        to_stop = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and not x.is_trading
            and self.get_trade_type_from_level_id(x.custom_info["level_id"]) == trade_type)
        return [StopExecutorAction(controller_id=self.config.id, executor_id=executor.id) for executor in to_stop]

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        trade_type = self.get_trade_type_from_level_id(level_id)
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )

    def get_candles_config(self) -> List[CandlesConfig]:
        return [CandlesConfig(
            connector=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )]
