import asyncio
import threading
from typing import TYPE_CHECKING, List

import aiohttp

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    MarketEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
)
from hummingbot.notifier.notifier_base import NotifierBase

if TYPE_CHECKING:
    from hummingbot.core.trading_core import TradingCore

TELEGRAM_API_URL = "https://api.telegram.org"


class TelegramNotifier(NotifierBase):
    """
    Outbound-only Telegram notifications: forwards messages sent via TradingCore.notify() (e.g. kill switch
    triggers) and reports order fills across the trading bot's connectors. Does not support inbound/two-way
    control (replying to the bot to issue commands) -- that's a deliberate scope boundary, not an oversight.
    """

    def __init__(self, token: str, chat_id: str, connectors: List[ConnectorBase], trading_core: "TradingCore"):
        super().__init__()
        self._token = token
        self._chat_id = chat_id
        self._connectors = connectors
        self._trading_core = trading_core
        self._ev_loop = asyncio.get_event_loop()

        self._fill_order_forwarder = SourceInfoEventForwarder(self._did_fill_order)
        self._event_pairs = [
            (MarketEvent.OrderFilled, self._fill_order_forwarder),
            (MarketEvent.BuyOrderCompleted, self._fill_order_forwarder),
            (MarketEvent.SellOrderCompleted, self._fill_order_forwarder),
        ]

    def start(self):
        for connector in self._connectors:
            for event_type, forwarder in self._event_pairs:
                connector.add_listener(event_type, forwarder)
        super().start()

    def stop(self):
        for connector in self._connectors:
            for event_type, forwarder in self._event_pairs:
                connector.remove_listener(event_type, forwarder)
        super().stop()

    def _did_fill_order(self, event_tag: int, market: ConnectorBase, evt: object):
        if threading.current_thread() != threading.main_thread():
            self._ev_loop.call_soon_threadsafe(self._did_fill_order, event_tag, market, evt)
            return

        if isinstance(evt, OrderFilledEvent):
            message = (f"[{market.name}] {evt.trade_type.name} {evt.amount} {evt.trading_pair} "
                       f"@ {evt.price} filled")
        elif isinstance(evt, (BuyOrderCompletedEvent, SellOrderCompletedEvent)):
            side = "BUY" if isinstance(evt, BuyOrderCompletedEvent) else "SELL"
            message = (f"[{market.name}] {side} order completed: {evt.base_asset_amount} "
                       f"{evt.base_asset}-{evt.quote_asset}")
        else:
            return
        self.add_message_to_queue(message)

    async def _send_message(self, message: str):
        url = f"{TELEGRAM_API_URL}/bot{self._token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session, session.post(
                url, data={"chat_id": self._chat_id, "text": message}
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    self.logger().warning(f"Telegram notification failed (status {resp.status}): {body}")
        except Exception:
            self.logger().warning("Error sending Telegram notification.", exc_info=True)
