from __future__ import annotations

import builtins
import contextlib
import io
import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RobinhoodOrderRequest:
    symbol: str
    side: str
    quantity: int
    limit_price: Optional[float] = None
    time_in_force: str = "gfd"


class RobinhoodClient:
    def __init__(self) -> None:
        self._logged_in = False
        self._rh = None

    def login(
        self,
        username: str,
        password: str,
        mfa_code: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ) -> bool:
        try:
            import robin_stocks.robinhood as rh
            import pyotp
        except Exception as exc:
            raise RuntimeError("robin-stocks package is not installed.") from exc

        if not username or not password:
            raise RuntimeError("Robinhood username and password are required.")

        resolved_code = (mfa_code or "").strip()
        resolved_secret = (totp_secret or "").replace(" ", "").strip()
        if resolved_secret and not resolved_code:
            try:
                resolved_code = pyotp.TOTP(resolved_secret).now()
            except Exception as exc:
                raise RuntimeError("Invalid Robinhood TOTP secret.") from exc

        pickle_name = re.sub(r"[^A-Za-z0-9]+", "_", username.lower()).strip("_")
        if not pickle_name:
            pickle_name = "default"

        self._rh = rh
        login_output = io.StringIO()

        @contextlib.contextmanager
        def verification_input_patch():
            original_input = builtins.input

            def patched_input(prompt: str = "") -> str:
                if resolved_code:
                    return resolved_code
                raise RuntimeError(
                    "Robinhood requested a verification code. Enter the current MFA/verification code or configure ROBINHOOD_TOTP_SECRET."
                )

            builtins.input = patched_input
            try:
                yield
            finally:
                builtins.input = original_input

        with verification_input_patch(), contextlib.redirect_stdout(login_output), contextlib.redirect_stderr(login_output):
            login_resp = rh.login(
                username=username,
                password=password,
                mfa_code=resolved_code or None,
                store_session=True,
                pickle_name=f"_{pickle_name}",
            )

        if not login_resp:
            details = login_output.getvalue().strip()
            if details:
                raise RuntimeError(f"Robinhood login failed. {details}")
            raise RuntimeError("Robinhood login failed. Check credentials, MFA code, approval prompt, or TOTP secret.")

        self._logged_in = bool(login_resp)
        return self._logged_in

    def profile_buying_power(self) -> Optional[float]:
        if not self._logged_in or self._rh is None:
            return None
        profile = self._rh.profiles.load_account_profile()
        if not profile:
            return None

        buying_power = profile.get("buying_power")
        if buying_power is None:
            margin_balances = profile.get("margin_balances") or {}
            buying_power = margin_balances.get("day_trade_buying_power") or margin_balances.get("overnight_buying_power")
        if buying_power is None:
            cash_available = profile.get("cash_available_for_withdrawal")
            buying_power = cash_available
        if buying_power is None:
            return None
        return float(buying_power)

    def submit_order(self, order: RobinhoodOrderRequest, paper_mode: bool = True) -> dict:
        if not self._logged_in or self._rh is None:
            raise RuntimeError("Robinhood session is not logged in.")

        if paper_mode:
            return {
                "status": "paper",
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "limit_price": order.limit_price,
            }

        if order.side.lower() == "buy":
            if order.limit_price:
                result = self._rh.orders.order_buy_limit(
                    order.symbol, order.quantity, order.limit_price, timeInForce=order.time_in_force
                )
            else:
                result = self._rh.orders.order_buy_market(
                    order.symbol, order.quantity, timeInForce=order.time_in_force
                )
        else:
            if order.limit_price:
                result = self._rh.orders.order_sell_limit(
                    order.symbol, order.quantity, order.limit_price, timeInForce=order.time_in_force
                )
            else:
                result = self._rh.orders.order_sell_market(
                    order.symbol, order.quantity, timeInForce=order.time_in_force
                )
        return result


def env_credentials() -> tuple[str, str, str]:
    return (
        os.getenv("ROBINHOOD_USERNAME", ""),
        os.getenv("ROBINHOOD_PASSWORD", ""),
        os.getenv("ROBINHOOD_TOTP_SECRET", ""),
    )
