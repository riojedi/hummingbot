from unittest import TestCase

from pydantic import ValidationError

from hummingbot.client.config.client_config_map import (
    ClientConfigMap,
    TelegramDisabledMode,
    TelegramEnabledMode,
)


class TelegramModeValidationTests(TestCase):
    """
    Mirrors the (untested elsewhere) kill_switch_mode validator's coercion rules -- telegram_mode uses the
    identical Union-mode dict/string/instance coercion pattern.
    """

    def test_defaults_to_disabled(self):
        cm = ClientConfigMap()
        self.assertIsInstance(cm.telegram_mode, TelegramDisabledMode)

    def test_accepts_enabled_mode_string(self):
        cm = ClientConfigMap(telegram_mode="telegram_enabled")
        self.assertIsInstance(cm.telegram_mode, TelegramEnabledMode)

    def test_accepts_disabled_mode_string(self):
        cm = ClientConfigMap(telegram_mode="telegram_disabled")
        self.assertIsInstance(cm.telegram_mode, TelegramDisabledMode)

    def test_rejects_unknown_string(self):
        with self.assertRaises(ValidationError):
            ClientConfigMap(telegram_mode="not_a_real_mode")

    def test_accepts_dict_with_credentials(self):
        cm = ClientConfigMap(telegram_mode={"telegram_token": "abc123", "telegram_chat_id": "42"})
        self.assertIsInstance(cm.telegram_mode, TelegramEnabledMode)
        self.assertEqual("42", cm.telegram_mode.telegram_chat_id)
        self.assertEqual("abc123", cm.telegram_mode.telegram_token.get_secret_value())

    def test_accepts_empty_dict_as_disabled(self):
        cm = ClientConfigMap(telegram_mode={})
        self.assertIsInstance(cm.telegram_mode, TelegramDisabledMode)

    def test_accepts_already_valid_instance(self):
        mode = TelegramEnabledMode(telegram_token="tok", telegram_chat_id="1")
        cm = ClientConfigMap(telegram_mode=mode)
        self.assertIs(cm.telegram_mode, mode)

    def test_rejects_unsupported_type(self):
        with self.assertRaises(ValidationError):
            ClientConfigMap(telegram_mode=12345)
