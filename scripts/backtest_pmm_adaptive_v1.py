"""
Backtest pmm_adaptive_v1 (NATR dynamic spreads + Avellaneda-Stoikov inventory skew) with optional chart output.

IMPORTANT LIMITATION: BacktestingEngineBase is driven entirely by historical candles and has no order book
depth history, so the third feature of this controller -- the order book imbalance (OBI) fill-kill in
`executors_to_early_stop` -- is NEVER exercised by this backtest. This script only validates the NATR
dynamic-spread and inventory-skew logic. Validate the OBI fill-kill path via
`test/hummingbot/strategy_v2/controllers/test_pmm_adaptive_v1.py` (synthetic order book snapshots) and via
paper trading (`<connector>_paper_trade`) before trusting it live.

Usage:
    conda run -n hummingbot python scripts/backtest_pmm_adaptive_v1.py
    conda run -n hummingbot python scripts/backtest_pmm_adaptive_v1.py --days 3 --chart
    conda run -n hummingbot python scripts/backtest_pmm_adaptive_v1.py --gamma 1.0 --target-pct 0.5
"""
import argparse
import asyncio
import os
import sys
import time

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch broken optional dependency (injective proto mismatch)
try:
    from pyinjective.proto.injective.stream.v2 import query_pb2
    if not hasattr(query_pb2, "OrderFailuresFilter"):
        query_pb2.OrderFailuresFilter = type("OrderFailuresFilter", (), {})
except ImportError:
    pass

from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase  # noqa: E402
from hummingbot.strategy_v2.backtesting.backtesting_result import BacktestingResult  # noqa: E402


def build_config(connector: str, trading_pair: str, total_amount_quote: int, gamma: float, target_pct: float):
    config_data = {
        "id": "backtest_pmm_adaptive_v1",
        "controller_name": "pmm_adaptive_v1",
        "controller_type": "market_making",
        "connector_name": connector,
        "trading_pair": trading_pair,
        "total_amount_quote": total_amount_quote,
        "leverage": 1,
        "buy_spreads": "1,2,4",
        "sell_spreads": "1,2,4",
        "candles_connector": connector,
        "candles_trading_pair": trading_pair,
        "interval": "3m",
        "natr_length": 14,
        "inventory_target_base_pct": str(target_pct),
        "risk_aversion": str(gamma),
        "max_skew_pct_of_natr": "0.5",
        "obi_depth_levels": 10,
        "obi_lookback_samples": 30,
        "obi_collapse_threshold_pct": "0.7",
        "obi_kill_switch_enabled": True,
        "stop_loss": "0.03",
        "take_profit": "0.02",
        "time_limit": 2700,
        "executor_refresh_time": 300,
        "cooldown_time": 15,
    }
    return BacktestingEngineBase.get_controller_config_instance_from_dict(
        config_data, controllers_module="controllers"
    )


async def main(days: float, show_chart: bool, output_path: str | None,
               connector: str, trading_pair: str, total_amount_quote: int,
               resolution: str, gamma: float, target_pct: float):
    end_ts = int(time.time())
    start_ts = end_ts - int(days * 24 * 3600)

    config = build_config(connector, trading_pair, total_amount_quote, gamma, target_pct)
    engine = BacktestingEngineBase()

    print(f"Running backtest: pmm_adaptive_v1 | {connector} {trading_pair} | {days}d | {resolution} "
          f"| gamma={gamma} target_pct={target_pct} ...")
    print("NOTE: this backtest does not exercise the OBI fill-kill path (no order book history) -- "
          "see the module docstring.")
    t0 = time.perf_counter()
    result = await engine.run_backtesting(
        config, start_ts, end_ts,
        backtesting_resolution=resolution,
        trade_cost=0.0002,
    )
    elapsed = time.perf_counter() - t0

    r = result["results"]
    n_candles = len(result["processed_data"].get("features", []))
    candles_per_sec = n_candles / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  pmm_adaptive_v1 backtest ({days}d @ {resolution})")
    print(f"{'=' * 60}")
    print(f"  Duration:               {elapsed:.2f}s ({n_candles} candles, {candles_per_sec:.0f} candles/s)")
    print(f"  Total executors:        {r['total_executors']}")
    print(f"  Net PnL:                {r['net_pnl_quote']:.4f} USDT ({r['net_pnl'] * 100:.2f}%)")
    print(f"  Accuracy:               {r['accuracy']:.2%}")
    print(f"  Sharpe ratio:           {r['sharpe_ratio']:.4f}")
    print(f"  Max drawdown:           {r['max_drawdown_pct']:.4%}")
    print(f"  Profit factor:          {r['profit_factor']:.4f}")
    print(f"  Close types:            {r['close_types']}")

    bt_result = BacktestingResult(result, config)
    print(f"\n{bt_result.get_results_summary()}")

    if show_chart:
        try:
            fig = bt_result.get_backtesting_figure()
            if output_path:
                fig.write_html(output_path)
                print(f"\n  Chart saved to {output_path}")
            else:
                fig.show()
        except ImportError:
            print("\n  plotly not installed: pip install plotly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest pmm_adaptive_v1")
    parser.add_argument("--days", type=float, default=0.5, help="Number of days to backtest (e.g. 0.5 for 12h)")
    parser.add_argument("--connector", type=str, default="binance")
    parser.add_argument("--trading-pair", type=str, default="BTC-USDT")
    parser.add_argument("--amount", type=int, default=1000, help="Total amount quote")
    parser.add_argument("--resolution", type=str, default="1m", help="Backtesting resolution (e.g. 1s, 1m, 5m)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Avellaneda-Stoikov risk aversion")
    parser.add_argument("--target-pct", type=float, default=0.5, help="Target base inventory fraction")
    parser.add_argument("--chart", action="store_true", default=True, help="Show/save the chart")
    parser.add_argument("--output", type=str, default=None, help="Save chart to HTML file instead of showing")
    args = parser.parse_args()

    asyncio.run(main(args.days, args.chart, args.output, args.connector, args.trading_pair, args.amount,
                     args.resolution, args.gamma, args.target_pct))
