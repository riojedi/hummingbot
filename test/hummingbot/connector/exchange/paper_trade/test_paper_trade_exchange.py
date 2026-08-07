from decimal import Decimal
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from hummingbot.connector.trading_rule import TradingRule

from hummingbot.connector.exchange.binance.binance_api_order_book_data_source import (
    BinanceAPIOrderBookDataSource,
)
from hummingbot.connector.exchange.kucoin.kucoin_api_order_book_data_source import (
    KucoinAPIOrderBookDataSource,
)
from hummingbot.connector.exchange.paper_trade import (
    create_paper_trade_market,
    get_order_book_tracker,
)
from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase


class PaperTradeExchangeTests(TestCase):

    def test_get_order_book_tracker_for_connector_using_generic_tracker(self):
        tracker = get_order_book_tracker(connector_name="binance", trading_pairs=["COINALPHA-HBOT"])
        self.assertEqual(OrderBookTracker, type(tracker))

        tracker = get_order_book_tracker(connector_name="kucoin", trading_pairs=["COINALPHA-HBOT"])
        self.assertEqual(OrderBookTracker, type(tracker))

    def test_create_paper_trade_market_for_connector_using_generic_tracker(self):
        paper_exchange = create_paper_trade_market(
            exchange_name="binance",
            trading_pairs=["COINALPHA-HBOT"])
        self.assertEqual(BinanceAPIOrderBookDataSource, type(paper_exchange.order_book_tracker.data_source))

        paper_exchange = create_paper_trade_market(
            exchange_name="kucoin",
            trading_pairs=["COINALPHA-HBOT"])
        self.assertEqual(KucoinAPIOrderBookDataSource, type(paper_exchange.order_book_tracker.data_source))


class PaperTradeExchangeTradingRulesTests(IsolatedAsyncioWrapperTestCase):
    """
    Regression coverage for the AttributeError: 'PaperTradeExchange' object has no attribute 'trading_rules'
    crash hit by every Strategy V2 executor (PositionExecutor, GridExecutor, TwapExecutor, DcaExecutor) when
    run against a *_paper_trade connector, since PaperTradeExchange never implemented trading_rules before.
    """

    def setUp(self):
        super().setUp()
        self.paper_exchange = create_paper_trade_market(
            exchange_name="binance",
            trading_pairs=["COINALPHA-HBOT"])

    def test_trading_rules_default_empty_before_start_network(self):
        self.assertEqual({}, self.paper_exchange.trading_rules)
        self.assertFalse(self.paper_exchange.ready)

    async def test_update_paper_trade_trading_rules_populates_dict(self):
        fake_rule = TradingRule(trading_pair="COINALPHA-HBOT", min_order_size=Decimal("0.0001"))
        fake_rules_connector = MagicMock()
        fake_rules_connector._update_trading_rules = AsyncMock()
        fake_rules_connector.trading_rules = {"COINALPHA-HBOT": fake_rule}
        fake_conn_setting = MagicMock()
        fake_conn_setting.non_trading_connector_instance_with_default_configuration = MagicMock(
            return_value=fake_rules_connector)

        with patch("hummingbot.client.settings.AllConnectorSettings.get_connector_settings",
                   return_value={"binance": fake_conn_setting}):
            await self.paper_exchange._update_paper_trade_trading_rules()

        self.assertEqual({"COINALPHA-HBOT": fake_rule}, self.paper_exchange.trading_rules)
        self.assertTrue(self.paper_exchange._trading_rules_initialized)

    async def test_update_paper_trade_trading_rules_graceful_on_failure(self):
        fake_conn_setting = MagicMock()
        fake_conn_setting.non_trading_connector_instance_with_default_configuration = MagicMock(
            side_effect=Exception("boom"))

        with patch("hummingbot.client.settings.AllConnectorSettings.get_connector_settings",
                   return_value={"binance": fake_conn_setting}):
            await self.paper_exchange._update_paper_trade_trading_rules()  # must not raise

        self.assertEqual({}, self.paper_exchange.trading_rules)
        self.assertTrue(self.paper_exchange._trading_rules_initialized)

    def test_ready_waits_for_trading_rules_initialization(self):
        with patch.object(type(self.paper_exchange.order_book_tracker), "ready",
                           new_callable=lambda: property(lambda self: True)):
            self.paper_exchange._trading_rules_initialized = False
            self.assertFalse(self.paper_exchange.ready)

            self.paper_exchange._trading_rules_initialized = True
            self.assertTrue(self.paper_exchange.ready)

    async def test_start_network_schedules_and_stop_network_cancels_trading_rules_update(self):
        with patch.object(
            type(self.paper_exchange), "_update_paper_trade_trading_rules", new_callable=AsyncMock
        ) as mock_update:
            await self.paper_exchange.start_network()
            self.assertIsNotNone(self.paper_exchange._trading_rules_update_task)
            await self.paper_exchange._trading_rules_update_task
            mock_update.assert_called_once()

            await self.paper_exchange.stop_network()
            self.assertIsNone(self.paper_exchange._trading_rules_update_task)
