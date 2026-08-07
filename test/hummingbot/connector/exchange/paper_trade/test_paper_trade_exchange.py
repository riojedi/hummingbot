import asyncio
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
from hummingbot.core.data_type.composite_order_book import CompositeOrderBook
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
        # `ready` also requires status_dict["order_books_initialized"], which needs at least one
        # populated order book (and, once true, `ready` calls init_paper_trade_market(), which
        # asserts each entry is a real CompositeOrderBook) -- set that up alongside the tracker's
        # own `ready` flag so this test isolates the new trading-rules gate instead of failing on
        # an unrelated precondition.
        with patch.object(type(self.paper_exchange.order_book_tracker), "ready",
                           new_callable=lambda: property(lambda self: True)), \
                patch.object(type(self.paper_exchange.order_book_tracker), "order_books",
                              new_callable=lambda: property(lambda self: {"COINALPHA-HBOT": CompositeOrderBook()})):
            self.paper_exchange._trading_rules_initialized = False
            self.assertFalse(self.paper_exchange.ready)

            self.paper_exchange._trading_rules_initialized = True
            self.assertTrue(self.paper_exchange.ready)

    async def test_start_network_schedules_and_stop_network_cancels_trading_rules_update(self):
        # PaperTradeExchange is a compiled `cdef class`: unlike a plain Python class, its methods
        # live in an immutable type slot and can't be `patch.object`'d at the class level (attempting
        # to do so raises "cannot set attribute of immutable type"). So instead of mocking
        # `_update_paper_trade_trading_rules` itself, mock what it calls internally -- the same
        # `AllConnectorSettings.get_connector_settings` seam used by the two tests above -- and hold
        # the coroutine open with an Event so there's a real, still-running task to cancel.
        # `order_book_tracker` is a plain Python object, so its `start`/`stop` (which would otherwise
        # open real network connections) can be mocked normally at the instance level.
        resume_event = asyncio.Event()

        async def _hang_until_resumed(*args, **kwargs):
            await resume_event.wait()

        fake_rules_connector = MagicMock()
        fake_rules_connector._update_trading_rules = AsyncMock(side_effect=_hang_until_resumed)
        fake_rules_connector.trading_rules = {}
        fake_conn_setting = MagicMock()
        fake_conn_setting.non_trading_connector_instance_with_default_configuration = MagicMock(
            return_value=fake_rules_connector)

        with patch.object(self.paper_exchange.order_book_tracker, "start"), \
                patch.object(self.paper_exchange.order_book_tracker, "stop"), \
                patch("hummingbot.client.settings.AllConnectorSettings.get_connector_settings",
                      return_value={"binance": fake_conn_setting}):
            await self.paper_exchange.start_network()
            update_task = self.paper_exchange._trading_rules_update_task
            self.assertIsNotNone(update_task)
            self.assertFalse(update_task.done())

            await self.paper_exchange.stop_network()
            self.assertIsNone(self.paper_exchange._trading_rules_update_task)
            # cancel() only *requests* cancellation; awaiting the task is what actually lets the
            # event loop deliver the CancelledError and settle it into the cancelled state.
            with self.assertRaises(asyncio.CancelledError):
                await update_task
            self.assertTrue(update_task.cancelled())

        resume_event.set()  # release the hung coroutine so it doesn't leak past the test
