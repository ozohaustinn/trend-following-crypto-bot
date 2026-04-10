"""Main orchestration logic — the daily execution loop."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

import bot_config
import config as strategy_config
from bot.exchange import ExchangeClient
from bot.logger import TradeLogger
from bot.notifier import format_daily_summary, send_telegram
from bot.positions import compute_order_deltas
from bot.risk import run_all_checks
from data.universe import select_universe
from strategy.ensemble import ensemble_weights


class BotRunner:
    """Orchestrate the full daily trading cycle.

    Parameters
    ----------
    mode:
        Operating mode: ``"dry"``, ``"testnet"``, or ``"live"``.
    strategy_mode:
        Strategy position mode: ``"long_only"`` or ``"long_short"``.
    n_assets:
        Number of assets in the portfolio universe.
    starting_equity:
        Simulated starting equity for dry-run mode (USDT).
    """

    def __init__(
        self,
        mode: str = "dry",
        strategy_mode: str = "long_only",
        n_assets: int = 20,
        starting_equity: float = 10_000.0,
    ) -> None:
        self.mode = mode
        self.strategy_mode = strategy_mode
        self.n_assets = n_assets
        self.exchange = ExchangeClient(mode=mode, starting_equity=starting_equity)
        self.logger = TradeLogger(log_dir=bot_config.LOG_DIR)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_ohlcv(self) -> dict[str, pd.DataFrame]:
        """Fetch ~450 days of daily OHLCV for all candidate symbols.

        Uses an incremental parquet cache: loads existing data, fetches
        only the new bars since the last cached date, and re-saves.
        """
        import ccxt
        from pathlib import Path

        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now(tz=timezone.utc).date()
        # 450 days of history to cover max lookback (360) + vol window (90)
        start = str(today - pd.Timedelta(days=450))
        end = str(today)

        all_data: dict[str, pd.DataFrame] = {}

        if self.mode == "dry":
            # In dry mode use a spot exchange for price data (no auth needed)
            ex = ccxt.binance()
        else:
            ex = ccxt.binanceusdm(
                {
                    "apiKey": (
                        bot_config.BINANCE_TESTNET_API_KEY
                        if self.mode == "testnet"
                        else bot_config.BINANCE_API_KEY
                    ),
                    "secret": (
                        bot_config.BINANCE_TESTNET_API_SECRET
                        if self.mode == "testnet"
                        else bot_config.BINANCE_API_SECRET
                    ),
                }
            )
            if self.mode == "testnet":
                ex.set_sandbox_mode(True)

        for symbol in bot_config.CANDIDATE_SYMBOLS:
            safe = symbol.replace("/", "_")
            cache_file = cache_dir / f"{safe}_live_cache.parquet"

            cached_df: pd.DataFrame | None = None
            if cache_file.exists():
                try:
                    cached_df = pd.read_parquet(cache_file)
                except Exception:  # noqa: BLE001
                    cached_df = None

            # Determine what to fetch
            if cached_df is not None and not cached_df.empty:
                last_cached = cached_df.index.max()
                fetch_start_ms = int(
                    (last_cached + pd.Timedelta(days=1)).timestamp() * 1000
                )
            else:
                fetch_start_ms = int(pd.Timestamp(start).timestamp() * 1000)

            end_ms = int(pd.Timestamp(end).timestamp() * 1000)

            new_rows: list[list] = []
            since_ms = fetch_start_ms
            while since_ms < end_ms:
                try:
                    batch = ex.fetch_ohlcv(
                        symbol, timeframe="1d", since=since_ms, limit=500
                    )
                except Exception:  # noqa: BLE001
                    break
                if not batch:
                    break
                new_rows.extend(batch)
                last_ts = batch[-1][0]
                if last_ts >= end_ms or len(batch) < 500:
                    break
                since_ms = last_ts + 86_400_000
                time.sleep(ex.rateLimit / 1000)

            if new_rows:
                new_df = pd.DataFrame(
                    new_rows,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                new_df["timestamp"] = pd.to_datetime(
                    new_df["timestamp"], unit="ms", utc=True
                )
                new_df = new_df.set_index("timestamp").sort_index()
                new_df = new_df[new_df.index < pd.Timestamp(end, tz="UTC")]
                new_df.index = new_df.index.normalize()

                if cached_df is not None and not cached_df.empty:
                    combined = pd.concat([cached_df, new_df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                else:
                    combined = new_df

                # Trim to the required history window
                cutoff_start = pd.Timestamp(start, tz="UTC")
                combined = combined[combined.index >= cutoff_start]

                try:
                    combined.to_parquet(cache_file)
                except Exception:  # noqa: BLE001
                    pass

                all_data[symbol] = combined
            elif cached_df is not None and not cached_df.empty:
                all_data[symbol] = cached_df
            else:
                all_data[symbol] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )

        return all_data

    def _select_universe(
        self, all_data: dict[str, pd.DataFrame], as_of_date: str
    ) -> list[str]:
        return select_universe(all_data, as_of_date, n_assets=self.n_assets)

    def _compute_target_weights(
        self,
        universe: list[str],
        all_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Run the ensemble strategy for each asset and return per-asset weights.

        The weight for each asset is scaled by ``1 / n_assets`` so total
        capital is divided equally across the universe.
        """
        target_weights: dict[str, float] = {}
        n = len(universe)

        for symbol in universe:
            df = all_data.get(symbol)
            if df is None or df.empty or "close" not in df.columns:
                continue

            close = df["close"].dropna()
            if len(close) < max(strategy_config.LOOKBACK_PERIODS):
                continue

            try:
                weights = ensemble_weights(
                    close=close,
                    lookback_periods=strategy_config.LOOKBACK_PERIODS,
                    mode=self.strategy_mode,
                    vol_target=strategy_config.VOL_TARGET,
                    vol_window=strategy_config.VOL_WINDOW,
                    max_leverage=strategy_config.MAX_LEVERAGE,
                    rebalance_threshold=strategy_config.REBALANCE_THRESHOLD,
                )
                asset_weight = float(weights.iloc[-1])
                # Equal capital allocation across universe
                target_weights[symbol] = asset_weight / n if n > 0 else 0.0
            except Exception:  # noqa: BLE001
                target_weights[symbol] = 0.0

        return target_weights

    def _get_last_prices(
        self,
        symbols: list[str],
        all_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Return the most recent close price for each symbol."""
        prices: dict[str, float] = {}
        for symbol in symbols:
            df = all_data.get(symbol)
            if df is not None and not df.empty and "close" in df.columns:
                prices[symbol] = float(df["close"].iloc[-1])
            else:
                prices[symbol] = self.exchange.get_current_price(symbol)
        return prices

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full daily cycle.

        Returns
        -------
        dict
            The structured run-data dict (also written to the daily log).
        """
        run_start = datetime.now(tz=timezone.utc)
        today_str = run_start.strftime("%Y-%m-%d")
        errors: list[str] = []

        # ---- Step 1: Fetch OHLCV ----------------------------------------
        try:
            all_data = self._fetch_ohlcv()
        except Exception as exc:  # noqa: BLE001
            all_data = {}
            errors.append(f"Data fetch failed: {exc}")

        # ---- Step 2: Select universe ------------------------------------
        try:
            universe = self._select_universe(all_data, today_str)
        except Exception as exc:  # noqa: BLE001
            universe = []
            errors.append(f"Universe selection failed: {exc}")

        # ---- Step 3: Compute target weights -----------------------------
        try:
            target_weights = self._compute_target_weights(universe, all_data)
        except Exception as exc:  # noqa: BLE001
            target_weights = {}
            errors.append(f"Weight computation failed: {exc}")

        # ---- Step 4: Fetch current positions ----------------------------
        try:
            if self.mode == "dry":
                # Simulate continuity by reading yesterday's logged targets
                current_weights = self.logger.get_last_positions()
            else:
                current_weights = self.exchange.get_positions()
        except Exception as exc:  # noqa: BLE001
            current_weights = {}
            errors.append(f"Position fetch failed: {exc}")

        # ---- Step 5: Compute order deltas --------------------------------
        order_deltas = compute_order_deltas(
            target_weights=target_weights,
            current_weights=current_weights,
            min_delta=bot_config.MIN_REBALANCE_DELTA,
        )

        # ---- Step 4b: Account equity ------------------------------------
        try:
            account_equity = self.exchange.get_account_equity()
        except Exception as exc:  # noqa: BLE001
            account_equity = 0.0
            errors.append(f"Equity fetch failed: {exc}")

        # ---- Step 5b: Last prices ----------------------------------------
        all_candidate_symbols = list(set(universe) | set(current_weights))
        prices = self._get_last_prices(all_candidate_symbols, all_data)

        # ---- Step 6 + 7: Risk checks & order execution ------------------
        orders_placed: list[dict] = []
        risk_summary = {
            "max_position_exceeded": False,
            "max_leverage_exceeded": False,
            "equity_floor_breach": False,
        }

        for order in order_deltas:
            symbol = order["symbol"]
            delta = order["delta_weight"]
            current_w = current_weights.get(symbol, 0.0)
            new_weight = current_w + delta
            price = prices.get(symbol, 1.0)
            order_value = abs(delta) * account_equity

            # Build the projected weights map for total-leverage check
            projected_weights = dict(current_weights)
            for sym, d in [(o["symbol"], o["delta_weight"]) for o in order_deltas]:
                projected_weights[sym] = projected_weights.get(sym, 0.0) + d

            approved, reason = run_all_checks(
                symbol=symbol,
                new_weight=new_weight,
                order_value_usd=order_value,
                all_new_weights=projected_weights,
                account_equity=account_equity,
                max_single_position=bot_config.MAX_SINGLE_POSITION,
                max_total_leverage=bot_config.MAX_TOTAL_LEVERAGE,
                max_order_pct=bot_config.MAX_SINGLE_ORDER_PCT,
                min_order_value=bot_config.MIN_ORDER_VALUE,
                equity_floor=bot_config.EQUITY_FLOOR,
            )

            if not approved:
                errors.append(f"[RISK BLOCKED] {symbol}: {reason}")
                if "equity" in reason.lower() and "floor" in reason.lower():
                    risk_summary["equity_floor_breach"] = True
                elif "leverage" in reason.lower() and "total" in reason.lower():
                    risk_summary["max_leverage_exceeded"] = True
                elif "exceed" in reason.lower():
                    risk_summary["max_position_exceeded"] = True
                continue

            order_size = order_value / price if price > 0 else 0.0
            side = order["side"]

            if self.mode != "dry":
                try:
                    self.exchange.set_margin_type(
                        symbol, bot_config.DEFAULT_MARGIN_TYPE
                    )
                    leverage = max(1, int(bot_config.MAX_SINGLE_POSITION))
                    self.exchange.set_leverage(symbol, leverage)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Leverage/margin setup failed for {symbol}: {exc}")

            try:
                fill = self.exchange.place_market_order(symbol, side, order_size)
                orders_placed.append(fill)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Order failed for {symbol}: {exc}")
                orders_placed.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "amount": order_size,
                        "fill_price": price,
                        "status": "failed",
                        "order_id": None,
                    }
                )

        # ---- Step 8: Verify fills ----------------------------------------
        if self.mode != "dry" and orders_placed:
            time.sleep(5)
            try:
                post_positions = self.exchange.get_positions()
                for order in orders_placed:
                    sym = order["symbol"]
                    expected_w = target_weights.get(sym, 0.0)
                    actual_w = post_positions.get(sym, 0.0)
                    if abs(actual_w - expected_w) > 0.05:
                        errors.append(
                            f"Fill discrepancy for {sym}: "
                            f"expected {expected_w:.3f}, actual {actual_w:.3f}"
                        )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Fill verification failed: {exc}")

        # ---- Step 9: Build signals summary for log ----------------------
        all_log_symbols = set(universe) | set(current_weights)
        signals_log: dict[str, dict] = {}
        for sym in all_log_symbols:
            tgt = target_weights.get(sym, 0.0)
            cur = current_weights.get(sym, 0.0)
            delta = tgt - cur
            if abs(delta) > bot_config.MIN_REBALANCE_DELTA:
                action = "BUY" if delta > 0 else "SELL"
            else:
                action = "HOLD"
            signals_log[sym] = {
                "target_weight": tgt,
                "current_weight": cur,
                "delta": delta,
                "action": action,
            }

        run_data = {
            "timestamp": run_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": self.mode,
            "account_equity": account_equity,
            "universe": universe,
            "signals": signals_log,
            "orders_placed": orders_placed,
            "risk_checks": risk_summary,
            "errors": errors,
        }

        # ---- Step 9: Log everything --------------------------------------
        self.logger.log_run(run_data)

        # ---- Step 10: Notify (optional) ---------------------------------
        try:
            summary = format_daily_summary(run_data)
            send_telegram(summary)
        except Exception:  # noqa: BLE001
            pass

        return run_data
