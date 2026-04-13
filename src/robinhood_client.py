from __future__ import annotations

import os
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

    def login(self, username: str, password: str, mfa_code: Optional[str] = None) -> bool:
        try:
            import robin_stocks.robinhood as rh
        except Exception as exc:
            raise RuntimeError("robin-stocks package is not installed.") from exc

        self._rh = rh
        login_resp = rh.login(
            username=username,
            password=password,
            mfa_code=mfa_code,
            store_session=False,
        )
        self._logged_in = bool(login_resp)
        return self._logged_in

    def profile_buying_power(self) -> Optional[float]:
        if not self._logged_in or self._rh is None:
            return None
        profile = self._rh.profiles.load_account_profile()
        return float(profile.get("buying_power", 0.0))

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


def env_credentials() -> tuple[str, str]:
    return os.getenv("ROBINHOOD_USERNAME", ""), os.getenv("ROBINHOOD_PASSWORD", "")
