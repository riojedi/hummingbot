from unittest.mock import MagicMock

from aioresponses import aioresponses

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    MarketEvent,
    OrderFilledEvent,
)
from hummingbot.notifier.telegram_notifier import TELEGRAM_API_URL, TelegramNotifier
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase


class TestTelegramNotifier(IsolatedAsyncioWrapperTestCase):
    def setUp(self):
        super().setUp()
        self.connector = MagicMock()
        self.connector.name = "binance"
        self.notifier = TelegramNotifier(
            token="test-token",
            chat_id="12345",
            connectors=[self.connector],
            trading_core=MagicMock(),
        )

    @aioresponses()
    async def test_send_message_posts_to_correct_url_and_payload(self, mock_api):
        url = f"{TELEGRAM_API_URL}/bottest-token/sendMessage"
        mock_api.post(url, status=200, payload={"ok": True})

        await self.notifier._send_message("hello world")

        mock_api.assert_called_once()
        found_request = False
        for (req_method, req_url), req_data in mock_api.requests.items():
            if str(req_url) == url and req_method == "POST":
                found_request = True
                self.assertEqual({"chat_id": "12345", "text": "hello world"}, req_data[0].kwargs.get("data"))
        self.assertTrue(found_request)

    @aioresponses()
    async def test_send_message_http_failure_does_not_raise(self, mock_api):
        url = f"{TELEGRAM_API_URL}/bottest-token/sendMessage"
        mock_api.post(url, status=401, body="Unauthorized")

        await self.notifier._send_message("hello world")  # must not raise

    async def test_send_message_connection_error_does_not_raise(self):
        with aioresponses() as mock_api:
            url = f"{TELEGRAM_API_URL}/bottest-token/sendMessage"
            mock_api.post(url, exception=ConnectionError("boom"))
            await self.notifier._send_message("hello world")  # must not raise

    def test_fill_event_queues_formatted_message(self):
        evt = OrderFilledEvent(
            timestamp=1234567890,
            order_id="OID-1",
            trading_pair="ETH-USDT",
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=100,
            amount=1,
            trade_fee=MagicMock(),
        )
        self.notifier._did_fill_order(MarketEvent.OrderFilled.value, self.connector, evt)

        self.assertEqual(1, self.notifier._message_queue.qsize())
        message = self.notifier._message_queue.get_nowait()
        self.assertIn("binance", message)
        self.assertIn("BUY", message)
        self.assertIn("ETH-USDT", message)

    def test_buy_order_completed_event_queues_formatted_message(self):
        evt = BuyOrderCompletedEvent(
            timestamp=1234567890,
            order_id="OID-1",
            base_asset="ETH",
            quote_asset="USDT",
            base_asset_amount=1,
            quote_asset_amount=100,
            order_type=OrderType.LIMIT,
        )
        self.notifier._did_fill_order(MarketEvent.BuyOrderCompleted.value, self.connector, evt)

        self.assertEqual(1, self.notifier._message_queue.qsize())
        message = self.notifier._message_queue.get_nowait()
        self.assertIn("binance", message)
        self.assertIn("BUY", message)

    def test_start_and_stop_register_and_unregister_listeners(self):
        self.notifier.start()
        self.assertEqual(3, self.connector.add_listener.call_count)

        self.notifier.stop()
        self.assertEqual(3, self.connector.remove_listener.call_count)
