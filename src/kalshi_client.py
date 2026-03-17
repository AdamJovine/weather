"""
Thin wrapper around the Kalshi Python SDK.

Install: pip install kalshi_python_sync

Kalshi SDK quickstart:
  https://trading-api.readme.io/reference/getting-started

Auth uses RSA key-pair. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config import KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_BASE_URL


def _build_kalshi_client():
    """
    Construct and return an authenticated Kalshi API client.

    KalshiClient is the high-level wrapper that bundles all API sub-clients.
    Auth is set via set_kalshi_auth(key_id, pem_content) — note the SDK takes
    the PEM file *content*, not the file path, despite the param name.
    """
    try:
        from kalshi_python_sync import KalshiClient, Configuration
    except ImportError as e:
        raise ImportError(
            "kalshi_python_sync not installed. Run: pip install kalshi_python_sync"
        ) from e

    private_key_path = Path(KALSHI_PRIVATE_KEY_PATH)
    if not private_key_path.exists():
        raise FileNotFoundError(f"Kalshi private key not found: {private_key_path}")

    pem_content = private_key_path.read_text()

    from kalshi_python_sync import KalshiAuth

    config = Configuration(host=KALSHI_BASE_URL)
    client = KalshiClient(configuration=config)
    # set_kalshi_auth has a NameError bug in the SDK; set the attribute directly
    client.kalshi_auth = KalshiAuth(KALSHI_API_KEY_ID, pem_content)
    return client


class KalshiWeatherClient:
    """
    Read and write access to Kalshi weather markets.

    Instantiate via KalshiWeatherClient.from_env() to pick up credentials from .env.
    """

    def __init__(self, raw_client):
        self._client = raw_client

    @classmethod
    def from_env(cls) -> "KalshiWeatherClient":
        return cls(_build_kalshi_client())

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def get_weather_markets(
        self,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 200,
    ) -> list[dict]:
        """
        Return open weather markets. Optionally filter by series_ticker.

        Kalshi weather series tickers follow a pattern like "HIGHNY" — inspect
        live markets to find the exact series tickers for cities you want.
        """
        params = dict(status=status, limit=limit)
        if series_ticker:
            params["series_ticker"] = series_ticker

        resp = self._client._market_api.get_markets(**params)
        markets = resp.markets if hasattr(resp, "markets") else resp.get("markets", [])

        # Normalize to plain dicts
        return [m if isinstance(m, dict) else m.to_dict() for m in markets]

    def get_market(self, ticker: str) -> dict:
        resp = self._client._market_api.get_market(ticker)
        market = resp.market if hasattr(resp, "market") else resp.get("market", resp)
        return market if isinstance(market, dict) else market.to_dict()

    def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        """
        Fetch the orderbook for a market.

        Returns a dict with yes/no bid and ask levels.
        """
        resp = self._client._market_api.get_market_orderbook(ticker, depth=depth)
        ob = resp.orderbook if hasattr(resp, "orderbook") else resp
        return ob if isinstance(ob, dict) else ob.to_dict()

    def get_best_prices(self, ticker: str) -> dict:
        """
        Return best yes_ask and no_ask in cents for a market.

        Falls back to fetching full market if orderbook unavailable.
        """
        ob = self.get_orderbook(ticker)
        # Orderbook schema: yes side asks sorted ascending, no side asks sorted ascending
        yes_asks = ob.get("yes", {}).get("asks", []) or ob.get("yes_asks", [])
        no_asks = ob.get("no", {}).get("asks", []) or ob.get("no_asks", [])

        yes_ask = yes_asks[0]["price"] if yes_asks else None
        no_ask = no_asks[0]["price"] if no_asks else None

        return {"yes_ask": yes_ask, "no_ask": no_ask}

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Portfolio — fills, settlements, positions
    # ------------------------------------------------------------------

    def get_fills(
        self,
        ticker: Optional[str] = None,
        limit: int = 200,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[dict]:
        """
        Return executed fills from your portfolio.

        Each fill has: ticker, side, count, yes_price, no_price, created_time, trade_id.
        Filter by ticker or timestamp range (Unix seconds) as needed.
        """
        params = dict(limit=limit)
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts

        resp = self._client._portfolio_api.get_fills(**params)
        fills = resp.fills if hasattr(resp, "fills") else resp.get("fills", [])
        return [f if isinstance(f, dict) else f.to_dict() for f in fills]

    def get_settlements(
        self,
        ticker: Optional[str] = None,
        limit: int = 200,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[dict]:
        """
        Return settled positions.

        Each entry has: ticker, revenue, settled_time, no_count, yes_count,
        market_result ("yes" | "no"), profit.
        revenue = net cash received after settlement.
        """
        params = dict(limit=limit)
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts

        resp = self._client._portfolio_api.get_settlements(**params)
        settlements = resp.settlements if hasattr(resp, "settlements") else resp.get("settlements", [])
        return [s if isinstance(s, dict) else s.to_dict() for s in settlements]

    def get_positions(
        self,
        ticker: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        """
        Return current open positions.

        Each position has: ticker, position (net yes contracts), total_traded,
        fees_paid, market_exposure, realized_pnl, resting_orders_count.
        """
        params = dict(limit=limit)
        if ticker:
            params["ticker"] = ticker

        resp = self._client._portfolio_api.get_positions(**params)
        positions = resp.market_positions if hasattr(resp, "market_positions") else resp.get("market_positions", [])
        return [p if isinstance(p, dict) else p.to_dict() for p in positions]

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: str,       # "yes" or "no"
        count: int,      # number of contracts (each contract = $1 max payout)
        price: int,      # limit price in cents
        action: str = "buy",
        order_type: str = "limit",
        client_order_id: Optional[str] = None,
    ) -> dict:
        """
        Place a limit order on Kalshi.

        Uses demo environment if KALSHI_BASE_URL points to demo.
        Always call with dry_run=True in strategy.py until you're confident.

        SDK docs: https://trading-api.readme.io/reference/createorder
        """
        kwargs = dict(
            ticker=ticker,
            action=action,
            side=side,
            count=count,
            type=order_type,
        )
        if side == "yes":
            kwargs["yes_price"] = price
        else:
            kwargs["no_price"] = price
        if client_order_id:
            kwargs["client_order_id"] = client_order_id

        resp = self._client._orders_api.create_order(**kwargs)
        return resp if isinstance(resp, dict) else resp.to_dict()
