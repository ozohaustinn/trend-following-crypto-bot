"""Fetch daily OHLCV data from exchanges via ccxt with local parquet caching."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str, exchange: str, start: str, end: str) -> Path:
    safe_symbol = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe_symbol}_{exchange}_{start}_{end}.parquet"


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    exchange: str = "binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV for a single symbol.

    Parameters
    ----------
    symbol:
        Trading pair in ccxt format, e.g. ``"BTC/USDT"``.
    start:
        Start date as ``"YYYY-MM-DD"`` string (inclusive).
    end:
        End date as ``"YYYY-MM-DD"`` string (exclusive).
    exchange:
        ccxt exchange id (default ``"binance"``).
    use_cache:
        When *True* (default) results are cached as parquet files and
        re-used on subsequent calls with identical parameters.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns: ``open, high, low, close, volume``.
    """
    cache_file = _cache_path(symbol, exchange, start, end)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    import ccxt  # imported lazily so the module is usable without network

    ex = getattr(ccxt, exchange)()
    since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)

    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe="1d", since=since_ms, limit=500)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= end_ms or len(batch) < 500:
            break
        since_ms = last_ts + 86_400_000  # advance one day
        time.sleep(ex.rateLimit / 1000)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < pd.Timestamp(end, tz="UTC")]
    df.index = df.index.normalize()  # strip time component

    if use_cache:
        df.to_parquet(cache_file)

    return df


def fetch_multiple(
    symbols: list[str],
    start: str,
    end: str,
    exchange: str = "binance",
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for multiple symbols.

    Parameters
    ----------
    symbols:
        List of trading pairs, e.g. ``["BTC/USDT", "ETH/USDT"]``.
    start, end:
        Date range strings (``"YYYY-MM-DD"``).
    exchange:
        ccxt exchange id (default ``"binance"``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{symbol: df}`` with the same structure as :func:`fetch_ohlcv`.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            result[symbol] = fetch_ohlcv(symbol, start, end, exchange)
        except Exception as exc:  # noqa: BLE001
            print(f"[loader] Failed to fetch {symbol}: {exc}")
    return result
