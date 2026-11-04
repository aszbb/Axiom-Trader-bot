"""ASZ AI FOREX TERMINAL - single-file Streamlit app (Forex only, free public data).

Data provider: biquote.io (documented, keyless REST API). No prices are ever simulated.
Predictions are statistical model estimates, NOT guarantees and NOT calibrated win rates.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK, PLOTLY_ERR = True, ""
except Exception as _e:  # optional dependency: UI degrades gracefully
    PLOTLY_OK, PLOTLY_ERR = False, str(_e)

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SK_OK, SK_ERR = True, ""
except Exception as _e:
    SK_OK, SK_ERR = False, str(_e)

BASE = "https://biquote.io"
HEADERS = {"User-Agent": "ASZ-AI-Forex-Terminal/1.0"}
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"]
# label -> (minutes, native provider interval, native minutes). 2m/3m aggregate from 1m; 10m from 5m.
TF_MAP = {
    "1 Minute": (1, "1m", 1), "2 Minutes": (2, "1m", 1), "3 Minutes": (3, "1m", 1),
    "5 Minutes": (5, "5m", 5), "10 Minutes": (10, "5m", 5), "15 Minutes": (15, "15m", 15),
}
WARM = 200            # bars consumed by the longest indicator lookback
MIN_BARS = WARM + 120


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    try:
        d = st.session_state.setdefault("diag", [])
        d.append(f"{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')} UTC | {msg}")
        del d[:-40]
    except Exception:
        pass


def try_call(fn, *args):
    """Run fn; return (value, None) or (None, error_text). Never raises."""
    try:
        return fn(*args), None
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        log(f"{getattr(fn, '__name__', 'call')} failed - {msg}")
        return None, msg


# ----------------------------------------------------------------------------
# Provider layer (biquote.io)
# ----------------------------------------------------------------------------
def _get(path: str, params=None):
    r = requests.get(BASE + path, params=params, headers=HEADERS, timeout=(5, 15))
    if r.status_code == 429:
        raise RuntimeError(f"Rate limited by provider (429). Retry after {r.headers.get('Retry-After', '60')}s")
    if r.status_code == 404:
        raise RuntimeError("404: unknown symbol or no quote in the last 7 days")
    if r.status_code == 503:
        raise RuntimeError("503: provider market-data store unreachable (provider outage)")
    r.raise_for_status()
    return r.json()


def _as_list(data) -> List[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("items", "symbols", "data", "ticks", "results"):
            if isinstance(data.get(k), list):
                return [x for x in data[k] if isinstance(x, dict)]
        if "symbol" in data or "name" in data:
            return [data]
        vals = [v for v in data.values() if isinstance(v, dict) and ("bid" in v or "symbol" in v)]
        return vals
    return []


def _num(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _norm_tick(t: dict) -> dict:
    bid, ask = _num(t.get("bid")), _num(t.get("ask"))
    mid = _num(t.get("mid"))
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2
    spread = _num(t.get("spread"))
    if spread is None and bid is not None and ask is not None:
        spread = ask - bid
    return {
        "symbol": t.get("symbol"), "bid": bid, "ask": ask, "mid": mid, "spread": spread,
        "high": _num(t.get("high")), "low": _num(t.get("low")),
        "dayDiffPercent": _num(t.get("dayDiffPercent")),
        "marketState": str(t.get("marketState", "unknown")), "stale": bool(t.get("stale", False)),
        "quoteAgeSeconds": _num(t.get("quoteAgeSeconds")),
        "lastQuoteAt": t.get("lastQuoteAt") or t.get("timestamp"), "source": t.get("source", ""),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_symbols() -> List[Tuple[str, str]]:
    rows = _as_list(_get("/api/symbols", {"type": "Forex", "quotedWithinDays": 7}))
    out = sorted({(str(r["name"]), str(r.get("description", ""))) for r in rows if r.get("name")})
    if not out:
        raise RuntimeError("Provider returned no Forex symbols (unexpected response shape?)")
    return out


@st.cache_data(ttl=3, show_spinner=False)
def fetch_tick(symbol: str) -> dict:
    t = _norm_tick(_get(f"/api/{symbol}"))
    if t["bid"] is None or t["ask"] is None:
        raise RuntimeError("Tick response has no bid/ask")
    return t


@st.cache_data(ttl=5, show_spinner=False)
def fetch_latest(symbols: Tuple[str, ...]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i in range(0, len(symbols), 40):
        chunk = list(symbols[i:i + 40])
        for row in _as_list(_get("/api/latest", {"symbols": chunk})):
            t = _norm_tick(row)
            if t["symbol"]:
                out[str(t["symbol"])] = t
    if not out:
        raise RuntimeError("No ticks returned for watchlist (unexpected response shape or closed feed)")
    return out


@st.cache_data(ttl=20, show_spinner=False)
def fetch_ohlc(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    data = _get(f"/api/{symbol}/ohlc", {"interval": interval, "limit": limit})
    bars = data.get("bars") if isinstance(data, dict) else data
    if not bars:
        raise RuntimeError(f"No {interval} candles returned for {symbol}")
    df = pd.DataFrame(bars)
    need = {"openTime", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"Unexpected OHLC fields: {list(df.columns)}")
    df["time"] = pd.to_datetime(df["openTime"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "tickVolume"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else 0.0
    if "isOpen" not in df.columns:
        df["isOpen"] = False
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "tickVolume", "isOpen"]]


def resample_ohlc(df: pd.DataFrame, minutes: int, base: int) -> pd.DataFrame:
    """Aggregate closed bars into `minutes` candles, keeping only COMPLETE buckets."""
    b = df["time"].dt.floor(f"{minutes}min")
    g = df.groupby(b).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                          close=("close", "last"), tickVolume=("tickVolume", "sum"), n=("close", "size"))
    g = g[g["n"] == minutes // base].drop(columns="n")
    g.index.name = "time"
    return g.reset_index()


def get_candles(symbol: str, tf_label: str) -> pd.DataFrame:
    minutes, interval, base = TF_MAP[tf_label]
    raw = fetch_ohlc(symbol, interval, 1000)
    closed = raw[~raw["isOpen"]].reset_index(drop=True)
    out = closed[["time", "open", "high", "low", "close", "tickVolume"]] if minutes == base \
        else resample_ohlc(closed, minutes, base)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Indicator primitives (all causal: value at t uses data <= t only)
# ----------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 * up / (up + dn).replace(0, np.nan)).fillna(50)


def true_range(h, l, c) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h, l, c, n: int) -> pd.Series:
    return true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()


def dmi(h, l, c, n: int):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    a = atr(h, l, c, n).replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi.fillna(0), mdi.fillna(0), dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def stoch(h, l, c, k: int, d: int):
    ll, hh = l.rolling(k).min(), h.rolling(k).max()
    kk = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).fillna(50)
    return kk, kk.rolling(d).mean().fillna(50)


def cci(h, l, c, n: int) -> pd.Series:
    tp = (h + l + c) / 3
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((tp - tp.rolling(n).mean()) / (0.015 * md.replace(0, np.nan))).fillna(0)


def willr(h, l, c, n: int) -> pd.Series:
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    return (-100 * (hh - c) / (hh - ll).replace(0, np.nan)).fillna(-50)


def psar_trend(h: pd.Series, l: pd.Series, af0: float, afmax: float) -> np.ndarray:
    hv, lv, n = h.values, l.values, len(h)
    tr = np.zeros(n)
    if n < 3:
        return tr
    up, sar, ep, af = True, lv[0], hv[0], af0
    tr[0] = 1
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, lv[i - 1], lv[i - 2] if i > 1 else lv[i - 1])
            if lv[i] < sar:
                up, sar, ep, af = False, ep, lv[i], af0
            elif hv[i] > ep:
                ep, af = hv[i], min(af + af0, afmax)
        else:
            sar = max(sar, hv[i - 1], hv[i - 2] if i > 1 else hv[i - 1])
            if hv[i] > sar:
                up, sar, ep, af = True, ep, hv[i], af0
            elif lv[i] < ep:
                ep, af = lv[i], min(af + af0, afmax)
        tr[i] = 1 if up else -1
    return tr


def ichimoku(h, l, t: int, k: int, sb: int):
    tenkan = (h.rolling(t).max() + l.rolling(t).min()) / 2
    kijun = (h.rolling(k).max() + l.rolling(k).min()) / 2
    a = ((tenkan + kijun) / 2).shift(k)
    b = ((h.rolling(sb).max() + l.rolling(sb).min()) / 2).shift(k)
    return tenkan, kijun, a, b


def reg_tstat(c: pd.Series, n: int) -> np.ndarray:
    x = np.arange(n, dtype=float)
    xm, sxx = x.mean(), ((x - x.mean()) ** 2).sum()

    def f(y):
        ym = y.mean()
        b = ((x - xm) * (y - ym)).sum() / sxx
        s = math.sqrt(((y - ym - b * (x - xm)) ** 2).sum() / (n - 2))
        return b / (s / math.sqrt(sxx)) if s > 0 else 0.0
    return c.rolling(n).apply(f, raw=True).fillna(0).values


def candle_patterns(o, h, l, c) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    po, pc_, ph, pl = (np.r_[np.nan, x[:-1]] for x in (o, c, h, l))
    body, rng = np.abs(c - o), h - l
    uw, lw = h - np.maximum(o, c), np.minimum(o, c) - l
    with np.errstate(invalid="ignore"):
        green, red = c > o, c < o
        pgreen, pred = pc_ > po, pc_ < po
        bull_eng = pred & green & (c >= po) & (o <= pc_)
        bear_eng = pgreen & red & (c <= po) & (o >= pc_)
        hammer = (rng > 0) & (lw >= 0.6 * rng) & (body <= 0.3 * rng)
        star = (rng > 0) & (uw >= 0.6 * rng) & (body <= 0.3 * rng)
        doji = (rng > 0) & (body <= 0.1 * rng)
        doji_bull, doji_bear = doji & pred, doji & pgreen
    s1 = lambda m: np.r_[[False], m[:-1]]  # noqa: E731
    s2 = lambda m: np.r_[[False, False], m[:-2]]  # noqa: E731
    cs1, cs2 = np.r_[np.nan, c[:-1]], np.r_[np.nan, np.nan, c[:-2]]
    with np.errstate(invalid="ignore"):
        soldiers = green & s1(green) & s2(green) & (c > cs1) & (cs1 > cs2)
        crows = red & s1(red) & s2(red) & (c < cs1) & (cs1 < cs2)
        ph2 = np.r_[np.nan, np.nan, h[:-2]]
        pl2 = np.r_[np.nan, np.nan, l[:-2]]
        inside_prev = (ph < ph2) & (pl > pl2)
        in_bull, in_bear = inside_prev & (c > ph), inside_prev & (c < pl)
    nz = lambda m: np.nan_to_num(m).astype(bool)  # noqa: E731
    return {
        "Engulfing": (nz(bull_eng), nz(bear_eng)), "Pin bar (hammer/star)": (nz(hammer), nz(star)),
        "Doji reversal": (nz(doji_bull), nz(doji_bear)), "Three soldiers/crows": (nz(soldiers), nz(crows)),
        "Inside-bar breakout": (nz(in_bull), nz(in_bear)),
    }


# ----------------------------------------------------------------------------
# Indicator research library (parameter configurations, NOT independent indicators)
# ----------------------------------------------------------------------------
def build_configs(df: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray]:
    o, h, l, c, tv = (df[k].astype(float).reset_index(drop=True) for k in ("open", "high", "low", "close", "tickVolume"))
    T = len(c)
    names: List[str] = []
    fams: List[str] = []
    rows: List[np.ndarray] = []

    def add(fam: str, name: str, sig) -> None:
        names.append(name)
        fams.append(fam)
        rows.append(np.sign(np.nan_to_num(np.asarray(sig, dtype=float))).astype(np.int8))

    def sg(bull, bear) -> np.ndarray:
        return np.asarray(bull, bool).astype(np.int8) - np.asarray(bear, bool).astype(np.int8)

    def persist(sig: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return sig
        s = pd.Series(sig.astype(float))
        return np.where(s.rolling(k).min() == 1, 1, np.where(s.rolling(k).max() == -1, -1, 0))

    memo: Dict[Tuple[str, int], np.ndarray] = {}

    def ma(kind: str, n: int) -> np.ndarray:
        if (kind, n) not in memo:
            memo[(kind, n)] = (sma(c, n) if kind == "SMA" else ema(c, n)).values
        return memo[(kind, n)]

    MODES = (("Trend", 1), ("Reversion", -1))
    cv, hv, lv, ov = c.values, h.values, l.values, o.values

    for n in range(2, 31):                                            # RSI
        r = rsi(c, n).values
        for lo in (15, 20, 25, 30, 35, 40):
            rev = sg(r < lo, r > 100 - lo)
            for mn, m in MODES:
                for k in (1, 3):
                    add("RSI", f"RSI({n}) {lo}/{100 - lo} {mn} persist{k}", persist(-rev * m, k))

    for kind in ("SMA", "EMA"):                                       # price vs MA
        for n in list(range(3, 61)) + [70, 80, 90, 100, 120, 150, 200]:
            a = ma(kind, n)
            for mn, m in MODES:
                add("SMA/EMA price", f"{kind}({n}) price-cross {mn}", sg(cv > a, cv < a) * m)

    fasts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30]  # MA crossovers
    slows = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150, 200]
    for kind in ("SMA", "EMA"):
        for f in fasts:
            for s in slows:
                if f >= s:
                    continue
                fa, sa = ma(kind, f), ma(kind, s)
                base = sg(fa > sa, fa < sa)
                for mn, m in MODES:
                    for k in (1, 3):
                        add("MA crossover", f"{kind}({f})x{kind}({s}) {mn} persist{k}", persist(base * m, k))

    for f in (3, 5, 6, 8, 10, 12, 15, 18):                            # MACD
        for s in (13, 15, 17, 21, 26, 30, 35, 40):
            if f >= s:
                continue
            line = ema(c, f) - ema(c, s)
            for sgn in (5, 9, 12):
                hist = (line - ema(line, sgn)).values
                base = sg(hist > 0, hist < 0)
                for mn, m in MODES:
                    for k in (1, 3):
                        add("MACD", f"MACD({f},{s},{sgn}) {mn} persist{k}", persist(base * m, k))

    for n in (8, 10, 12, 14, 17, 20, 25, 30, 40, 50):                 # Bollinger
        mid, sd = sma(c, n), c.rolling(n).std()
        for k in (1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3):
            rev = sg(cv < (mid - k * sd).values, cv > (mid + k * sd).values)
            for mn, m in MODES:
                add("Bollinger", f"BB({n},{k}) {mn}", -rev * m)

    for k in (3, 5, 7, 9, 11, 14, 18, 21, 26, 30):                    # Stochastic
        for d in (2, 3, 5):
            _, dd = stoch(h, l, c, k, d)
            dv = dd.values
            for os_ in (10, 15, 20, 25, 30):
                rev = sg(dv < os_, dv > 100 - os_)
                for mn, m in MODES:
                    add("Stochastic", f"Stoch({k},{d}) {os_}/{100 - os_} {mn}", -rev * m)

    for n in (5, 7, 9, 10, 14, 20, 30, 40, 50):                       # CCI
        cc = cci(h, l, c, n).values
        for thr in (50, 75, 100, 150, 200):
            rev = sg(cc < -thr, cc > thr)
            for mn, m in MODES:
                add("CCI", f"CCI({n}) +/-{thr} {mn}", -rev * m)

    for n in (5, 7, 9, 10, 14, 18, 21, 28, 34, 50):                   # Williams %R
        w = willr(h, l, c, n).values
        for thr in (10, 15, 20, 25, 30):
            rev = sg(w < -100 + thr, w > -thr)
            for mn, m in MODES:
                add("Williams %R", f"WillR({n}) {thr} {mn}", -rev * m)

    for n in range(1, 41):                                            # ROC
        roc = c.pct_change(n)
        for sm in (1, 3, 5):
            r = (roc.rolling(sm).mean() if sm > 1 else roc).values
            for mn, m in MODES:
                add("ROC", f"ROC({n}) sm{sm} {mn}", sg(r > 0, r < 0) * m)

    for n in (3, 5, 8, 10, 14, 20, 30):                               # Momentum vs its average
        mom = c.diff(n)
        for sm in (3, 5, 9):
            mv, av = mom.values, mom.rolling(sm).mean().values
            for mn, m in MODES:
                add("Momentum", f"Mom({n}) vs SMA{sm} {mn}", sg(mv > av, mv < av) * m)

    for n in (5, 7, 10, 14, 20, 28):                                  # ADX / DI
        pdi, mdi, adx = (x.values for x in dmi(h, l, c, n))
        for amin in (0, 15, 20, 25, 30):
            tr = sg((pdi > mdi) & (adx >= amin), (mdi > pdi) & (adx >= amin))
            for mn, m in MODES:
                add("ADX/DI", f"DI({n}) ADX>={amin} {mn}", tr * m)

    for af0 in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):          # Parabolic SAR
        for afm in (0.1, 0.2, 0.3):
            tr = psar_trend(h, l, af0, afm)
            for mn, m in MODES:
                add("Parabolic SAR", f"SAR({af0},{afm}) {mn}", tr * m)

    for t in (5, 7, 9, 12):                                           # Ichimoku
        for k in (13, 17, 22, 26, 34):
            for sb in (26, 34, 52):
                tk, kj, a, b = (x.values for x in ichimoku(h, l, t, k, sb))
                add("Ichimoku", f"Ichi({t},{k},{sb}) TK-cross", sg(tk > kj, tk < kj))
                top, bot = np.fmax(a, b), np.fmin(a, b)
                add("Ichimoku", f"Ichi({t},{k},{sb}) cloud", sg(cv > top, cv < bot))

    for n in (5, 8, 10, 12, 15, 20, 30, 40, 50, 75, 100, 150):        # Support/resistance breakout
        hh, ll = h.shift(1).rolling(n).max().values, l.shift(1).rolling(n).min().values
        br = sg(cv > hh, cv < ll)
        add("Support/Resistance", f"Donchian({n}) breakout", br)
        add("Support/Resistance", f"Donchian({n}) fade", -br)

    for w in (10, 20, 30, 50, 100):                                   # Tick volume
        z = ((tv - tv.rolling(w).mean()) / tv.rolling(w).std().replace(0, np.nan)).fillna(0).values
        color = np.sign(cv - ov)
        for thr in (0.5, 1, 1.5, 2, 2.5, 3):
            for mn, m in (("Follow", 1), ("Fade", -1)):
                add("Tick volume", f"TickVol z({w})>{thr} {mn}", color * (z > thr) * m)
    signed = pd.Series(np.sign(cv - ov) * tv.values)
    for n in (5, 10, 20, 40):
        flow = signed.rolling(n).sum()
        for sm in (10, 20, 40):
            fv, av = flow.values, flow.rolling(sm).mean().values
            for mn, m in MODES:
                add("Tick volume", f"TickFlow({n}) vs SMA{sm} {mn}", sg(fv > av, fv < av) * m)

    pats = candle_patterns(ov, hv, lv, cv)                            # Candlestick patterns
    for pn, (pb, ps) in pats.items():
        add("Candlestick", f"{pn} raw", sg(pb, ps))
        add("Candlestick", f"{pn} inverted", -sg(pb, ps))
        for n in (10, 20, 50, 100):
            e = ema(c, n).values
            add("Candlestick", f"{pn} with-trend EMA{n}", sg(pb & (cv > e), ps & (cv < e)))
            add("Candlestick", f"{pn} counter-trend EMA{n}", sg(pb & (cv < e), ps & (cv > e)))

    for n in (10, 20, 30, 50):                                        # Keltner / volatility
        e = ema(c, n)
        for an in (7, 10, 14, 20):
            a = atr(h, l, c, an)
            for k in (0.5, 1, 1.5, 2, 2.5):
                br = sg(cv > (e + k * a).values, cv < (e - k * a).values)
                add("Volatility (Keltner/ATR)", f"Keltner({n},{an},{k}) breakout", br)
                add("Volatility (Keltner/ATR)", f"Keltner({n},{an},{k}) fade", -br)

    for n in (5, 8, 10, 15, 20, 30, 40, 50, 75, 100):                 # Trend strength
        ts = reg_tstat(c, n)
        for thr in (0, 1, 2):
            for mn, m in MODES:
                add("Trend strength", f"RegSlope t({n})>{thr} {mn}", sg(ts > thr, ts < -thr) * m)

    for n in (24, 48, 96, 144, 200, 288):                             # Pivot points (rolling window)
        piv = ((h.rolling(n).max() + l.rolling(n).min() + c) / 3).shift(1).values
        for mn, m in MODES:
            add("Pivot points", f"RollingPivot({n}) {mn}", sg(cv > piv, cv < piv) * m)

    for n in (50, 100, 200):                                          # Fibonacci levels
        hi, lo = h.rolling(n).max().shift(1).values, l.rolling(n).min().shift(1).values
        for lvl in (0.236, 0.382, 0.5, 0.618, 0.786):
            px = hi - lvl * (hi - lo)
            for mn, m in MODES:
                add("Fibonacci", f"Fib({n}) {lvl} {mn}", sg(cv > px, cv < px) * m)

    S = np.vstack(rows) if rows else np.zeros((0, T), np.int8)
    return names, fams, S


def wilson_lb(h, n, z: float = 1.645):
    n, h = np.asarray(n, float), np.asarray(h, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = h / n
        d = 1 + z * z / n
        out = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return np.where(n > 0, out, 0.0)


def evaluate(S: np.ndarray, o: np.ndarray, c: np.ndarray, lo: int, hi: int):
    """Signal at bar t (close) is scored against next candle direction sign(close[t+1]-open[t+1])."""
    ret = np.full(len(c), np.nan)
    ret[:-1] = c[1:] - o[1:]
    sgn = np.sign(ret)
    seg = np.zeros(len(c), bool)
    seg[lo:hi] = True
    m = np.isfinite(sgn) & (sgn != 0) & seg
    s0 = np.where(m, sgn, 0).astype(np.int8)
    n = ((S != 0) & m).sum(axis=1)
    hits = ((S * s0) == 1).sum(axis=1)
    return n, hits


@st.cache_data(ttl=300, show_spinner="Evaluating indicator configurations...")
def run_lab(df: pd.DataFrame, split: int, warm: int, topn: int = 30, min_sig: int = 30) -> dict:
    t0 = time.time()
    names, fams, S = build_configs(df)
    o, c = df["open"].values.astype(float), df["close"].values.astype(float)
    n_is, h_is = evaluate(S, o, c, warm, split)          # selection window (in-sample)
    n_os, h_os = evaluate(S, o, c, split, len(c) - 1)    # held-out window (never used to select)
    wl = wilson_lb(h_is, n_is)
    ok = n_is >= min_sig
    order = np.argsort(-np.where(ok, wl, -1.0))
    sel = [int(i) for i in order[:topn] if ok[i]]
    cons = S[sel].mean(axis=0) if sel else np.zeros(len(c))
    hr = lambda h, n: np.where(n > 0, h / np.maximum(n, 1), np.nan)  # noqa: E731
    tab = pd.DataFrame({
        "Family": fams, "Configuration": names, "IS signals": n_is, "IS hit %": hr(h_is, n_is) * 100,
        "IS Wilson LB %": wl * 100, "OOS signals": n_os, "OOS hit %": hr(h_os, n_os) * 100,
    })
    tab["Used by engine"] = False
    tab.loc[sel, "Used by engine"] = True
    top = tab[ok].sort_values("IS Wilson LB %", ascending=False).head(150).reset_index(drop=True)
    top.insert(0, "Rank", np.arange(1, len(top) + 1))
    g = tab[ok & (tab["OOS signals"] >= min_sig)].groupby("Family").agg(
        Configs=("Configuration", "size"), Median_IS_hit=("IS hit %", "median"), Median_OOS_hit=("OOS hit %", "median"))
    oos_pos = float((tab.loc[ok & (tab["OOS signals"] >= min_sig), "OOS hit %"] > 50).mean() * 100) if ok.any() else float("nan")
    sel_n, sel_h = (n_os[sel].sum(), h_os[sel].sum()) if sel else (0, 0)
    return {
        "n_cfg": len(names), "n_fam": len(set(fams)), "n_eligible": int(ok.sum()), "top": top, "families": g.reset_index(), "sel": sel,
        "sel_names": [names[i] for i in sel], "cons": cons, "elapsed": time.time() - t0,
        "sel_oos_hit": (sel_h / sel_n * 100) if sel_n else float("nan"), "sel_oos_n": int(sel_n),
        "pct_oos_above_50": oos_pos, "split": split, "warm": warm, "min_sig": min_sig,
    }


# ----------------------------------------------------------------------------
# Machine learning (chronological split, 1-bar purge)
# ----------------------------------------------------------------------------
FEATURE_COLS = ["r1", "r2", "r3", "r5", "r10", "rsi14", "rsi5", "macdh", "bbp", "atrp", "atr_ratio", "adx", "di",
                "stoch", "ema_sp", "ema50d", "tvz", "body", "uwick", "lwick", "hour_s", "hour_c"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, tv = (df[k].astype(float) for k in ("open", "high", "low", "close", "tickVolume"))
    a = atr(h, l, c, 14).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10):
        f[f"r{k}"] = c.diff(k) / a
    f["rsi14"], f["rsi5"] = rsi(c, 14), rsi(c, 5)
    macd = ema(c, 12) - ema(c, 26)
    f["macdh"] = (macd - ema(macd, 9)) / a
    mid, sd = sma(c, 20), c.rolling(20).std()
    f["bbp"] = (c - mid) / (2 * sd.replace(0, np.nan))
    f["atrp"] = a / c * 1e4
    f["atr_ratio"] = a / a.rolling(50).mean()
    pdi, mdi, adx = dmi(h, l, c, 14)
    f["adx"], f["di"] = adx, pdi - mdi
    f["stoch"] = stoch(h, l, c, 14, 3)[0]
    f["ema_sp"] = (ema(c, 9) - ema(c, 21)) / a
    f["ema50d"] = (c - ema(c, 50)) / a
    f["tvz"] = (tv - tv.rolling(30).mean()) / tv.rolling(30).std().replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["uwick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    f["lwick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    hr = df["time"].dt.hour + df["time"].dt.minute / 60.0
    f["hour_s"], f["hour_c"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    return f[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)


def _models():
    return [
        Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=0.3, max_iter=500))]),
        ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_leaf=15, random_state=7, n_jobs=1),
    ]


def _proba(models, X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


@st.cache_data(ttl=300, show_spinner="Training models (chronological split)...")
def run_ml(df: pd.DataFrame, split: int, warm: int) -> dict:
    if not SK_OK:
        return {"ok": False, "msg": f"scikit-learn unavailable: {SK_ERR}"}
    X = np.nan_to_num(make_features(df).values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    o, c = df["open"].values.astype(float), df["close"].values.astype(float)
    T = len(df)
    y = np.full(T, np.nan)
    r = c[1:] - o[1:]
    y[:-1] = np.where(r > 0, 1.0, np.where(r < 0, 0.0, np.nan))
    tr = np.arange(warm, split - 1)       # purge row split-1 (its label uses the first test bar)
    te = np.arange(split, T - 1)
    tr, te = tr[np.isfinite(y[tr])], te[np.isfinite(y[te])]
    if len(tr) < 150 or len(te) < 50 or len(np.unique(y[tr])) < 2:
        return {"ok": False, "msg": f"Not enough labeled data (train={len(tr)}, test={len(te)}; need >=150 / >=50)."}
    models = _models()
    for m in models:
        m.fit(X[tr], y[tr])
    p_te, y_te = _proba(models, X[te]), y[te]
    acc = float(np.mean((p_te >= 0.5) == (y_te == 1)))
    maj = 1.0 if y[tr].mean() >= 0.5 else 0.0
    base = float(np.mean(y_te == maj))
    pc = np.clip(p_te, 1e-6, 1 - 1e-6)
    n = len(te)
    z = (acc - 0.5) / math.sqrt(0.25 / n)
    bins = np.digitize(p_te, [0.45, 0.5, 0.55])
    cal = [{"Bin": lab, "n": int((bins == i).sum()), "Mean P(up)": float(p_te[bins == i].mean()) if (bins == i).any() else np.nan,
            "Observed up %": float(y_te[bins == i].mean() * 100) if (bins == i).any() else np.nan}
           for i, lab in enumerate(["<0.45", "0.45-0.50", "0.50-0.55", ">=0.55"])]
    imp = pd.Series(models[1].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    al = np.arange(warm, T - 1)
    al = al[np.isfinite(y[al])]
    fin = _models()
    for m in fin:
        m.fit(X[al], y[al])
    p_test = np.full(T, np.nan)
    p_test[te] = p_te
    return {
        "ok": True, "msg": "ok", "p_live": float(_proba(fin, X[[T - 1]])[0]), "p_test": p_test,
        "acc": acc, "base_acc": base, "brier": float(np.mean((p_te - y_te) ** 2)),
        "logloss": float(-np.mean(y_te * np.log(pc) + (1 - y_te) * np.log(1 - pc))),
        "n_train": int(len(tr)), "n_test": int(n), "z": float(z), "edge": bool(z > 1.645),
        "calibration": pd.DataFrame(cal), "importance": imp, "last_x": dict(zip(FEATURE_COLS, X[T - 1])),
    }


# ----------------------------------------------------------------------------
# Prediction + backtest
# ----------------------------------------------------------------------------
W_ML, W_IND = 0.6, 0.4


def blend(p_ml, p_ind):
    return np.where(np.isfinite(p_ml), W_ML * p_ml + W_IND * p_ind, p_ind)


def backtest(o, c, p, lo, hi, thr, spread) -> dict:
    t = np.arange(lo, hi)
    t = t[np.isfinite(p[t])]
    d = np.where(p[t] >= thr, 1, np.where(p[t] <= 1 - thr, -1, 0))
    ret = c[t + 1] - o[t + 1]
    gross = d * ret
    net = np.where(d != 0, gross - spread, 0.0)
    frac = net / o[t + 1]
    tr = d != 0
    n = int(tr.sum())
    eq = np.cumsum(frac)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:] if len(eq) else eq
    wins, losses = net[tr & (net > 0)].sum(), -net[tr & (net < 0)].sum()
    return {
        "bars": int(len(t)), "signals": n, "no_trade": int(len(t) - n), "buys": int((d == 1).sum()), "sells": int((d == -1).sum()),
        "hit_rate": float((gross[tr] > 0).mean() * 100) if n else float("nan"),
        "net_win_rate": float((net[tr] > 0).mean() * 100) if n else float("nan"),
        "wins": int((tr & (net > 0)).sum()), "losses": int((tr & (net < 0)).sum()),
        "profit_factor": float(wins / losses) if losses > 0 else float("nan"),
        "total_bps": float(frac.sum() * 1e4), "max_dd_bps": float((peak - eq).max() * 1e4) if len(eq) else 0.0,
        "eq": eq * 1e4, "t": t[: len(eq)],
    }


# ----------------------------------------------------------------------------
# Technical snapshot
# ----------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["close"], d["high"], d["low"]
    for n in (9, 21, 50):
        d[f"ema{n}"] = ema(c, n)
    d["sma20"], d["sma100"] = sma(c, 20), sma(c, 100)
    mid, sd = sma(c, 20), c.rolling(20).std()
    d["bb_u"], d["bb_l"] = mid + 2 * sd, mid - 2 * sd
    d["rsi14"] = rsi(c, 14)
    d["macd"] = ema(c, 12) - ema(c, 26)
    d["macds"] = ema(d["macd"], 9)
    d["macdh"] = d["macd"] - d["macds"]
    d["atr14"] = atr(h, l, c, 14)
    return d


def tech_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    d = add_indicators(df)
    h, l, c, tv = d["high"], d["low"], d["close"], d["tickVolume"]
    x = d.iloc[-1]
    px = float(x["close"])
    pdi, mdi, adx = (s.iloc[-1] for s in dmi(h, l, c, 14))
    k, dd = (s.iloc[-1] for s in stoch(h, l, c, 14, 3))
    cc, wr = cci(h, l, c, 20).iloc[-1], willr(h, l, c, 14).iloc[-1]
    roc, mom = c.pct_change(10).iloc[-1] * 100, c.diff(10).iloc[-1]
    sar = psar_trend(h, l, 0.02, 0.2)[-1]
    tk, kj, a, b = (s.iloc[-1] for s in ichimoku(h, l, 9, 26, 52))
    bbp = (px - x["bb_l"]) / (x["bb_u"] - x["bb_l"]) if x["bb_u"] != x["bb_l"] else 0.5

    def rd(v, bull, bear):
        return "Bullish" if bull else ("Bearish" if bear else "Neutral")
    rows = [
        ("RSI(14)", f"{x['rsi14']:.1f}", "Oversold (bullish bias)" if x["rsi14"] < 30 else "Overbought (bearish bias)" if x["rsi14"] > 70 else rd(0, x["rsi14"] > 52, x["rsi14"] < 48)),
        ("MACD hist", f"{x['macdh']:.6f}", rd(0, x["macdh"] > 0, x["macdh"] < 0)),
        ("EMA 9/21/50 stack", f"{x['ema9']:.5f} / {x['ema21']:.5f} / {x['ema50']:.5f}", rd(0, x["ema9"] > x["ema21"] > x["ema50"], x["ema9"] < x["ema21"] < x["ema50"])),
        ("Price vs SMA100", f"{px - x['sma100']:+.5f}" if np.isfinite(x["sma100"]) else "n/a", rd(0, px > x["sma100"], px < x["sma100"])),
        ("ADX(14) / +DI / -DI", f"{adx:.1f} / {pdi:.1f} / {mdi:.1f}", rd(0, pdi > mdi and adx > 20, mdi > pdi and adx > 20)),
        ("Stochastic(14,3) %K/%D", f"{k:.1f} / {dd:.1f}", "Oversold (bullish bias)" if k < 20 else "Overbought (bearish bias)" if k > 80 else rd(0, k > dd, k < dd)),
        ("CCI(20)", f"{cc:.1f}", "Oversold (bullish bias)" if cc < -100 else "Overbought (bearish bias)" if cc > 100 else "Neutral"),
        ("Williams %R(14)", f"{wr:.1f}", "Oversold (bullish bias)" if wr < -80 else "Overbought (bearish bias)" if wr > -20 else "Neutral"),
        ("ROC(10) %", f"{roc:.4f}", rd(0, roc > 0, roc < 0)),
        ("Momentum(10)", f"{mom:.6f}", rd(0, mom > 0, mom < 0)),
        ("Parabolic SAR(0.02,0.2)", "uptrend" if sar > 0 else "downtrend", "Bullish" if sar > 0 else "Bearish"),
        ("Ichimoku(9,26,52)", f"TK {tk:.5f} / KJ {kj:.5f}", rd(0, tk > kj and px > max(a, b), tk < kj and px < min(a, b)) if np.isfinite(a) and np.isfinite(b) else "n/a"),
        ("Bollinger %B(20,2)", f"{bbp:.2f}", "Below lower band" if bbp < 0 else "Above upper band" if bbp > 1 else "Inside bands"),
        ("ATR(14)", f"{x['atr14']:.6f}", f"{x['atr14'] / px * 1e4:.1f} bps of price"),
    ]
    t = pd.DataFrame(rows, columns=["Indicator", "Value", "Reading"])
    v = t["Reading"].str.contains("Bullish|bullish").sum() - t["Reading"].str.contains("Bearish|bearish").sum()
    z = ((tv.iloc[-1] - tv.tail(50).mean()) / tv.tail(50).std()) if tv.tail(50).std() > 0 else 0.0
    info = {"net_votes": int(v), "n": len(t), "atr": float(x["atr14"]), "tv_z": float(z), "px": px, "d": d}
    return t, info


def sr_levels(df: pd.DataFrame, px: float, k: int = 3, look: int = 150):
    d = df.tail(look).reset_index(drop=True)
    hh, ll = d["high"].values, d["low"].values
    ph = [hh[i] for i in range(k, len(d) - k) if hh[i] == hh[i - k:i + k + 1].max()]
    pl = [ll[i] for i in range(k, len(d) - k) if ll[i] == ll[i - k:i + k + 1].min()]
    res = min([v for v in ph if v > px], default=float(hh.max()))
    sup = max([v for v in pl if v < px], default=float(ll.min()))
    hi, lo = float(hh.max()), float(ll.min())
    fib = {f"{p:.3f}": hi - p * (hi - lo) for p in (0.236, 0.382, 0.5, 0.618, 0.786)}
    pv = (hi + lo + float(d["close"].iloc[-1])) / 3
    return sup, res, fib, {"P": pv, "R1": 2 * pv - lo, "S1": 2 * pv - hi}


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@400;600&display=swap');
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#1a0b3a 0%,transparent 60%),radial-gradient(900px 500px at 100% 0%,#062a4a 0%,transparent 55%),#070a14;color:#e6f1ff;font-family:'Inter',sans-serif}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1024,#120a2e);border-right:1px solid #1f2a5a}
.asz-title{font-family:'Orbitron',sans-serif;font-weight:800;font-size:1.9rem;letter-spacing:.08em;background:linear-gradient(90deg,#00f0ff,#4d7cff,#b14dff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.asz-sub{color:#7f8fb8;font-size:.8rem;letter-spacing:.12em;margin-bottom:.6rem}
.card{background:linear-gradient(145deg,rgba(18,26,60,.85),rgba(20,12,48,.85));border:1px solid #243069;border-radius:14px;padding:.8rem 1rem;box-shadow:0 0 18px rgba(0,240,255,.07);min-height:92px}
.card .lbl{color:#7f8fb8;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase}
.card .val{font-family:'Orbitron',sans-serif;font-size:1.45rem;color:#00f0ff;margin-top:.15rem}
.card .sub{color:#9fb0d8;font-size:.75rem;margin-top:.1rem}
.card.buy{border-color:#00e676;box-shadow:0 0 22px rgba(0,230,118,.25)} .card.buy .val{color:#00e676}
.card.sell{border-color:#ff1744;box-shadow:0 0 22px rgba(255,23,68,.25)} .card.sell .val{color:#ff1744}
.card.wait .val{color:#ffd54f} .card.warn{border-color:#ffb300} .card.warn .val{color:#ffb300}
h2,h3{font-family:'Orbitron',sans-serif;color:#9bd8ff;letter-spacing:.06em}
.note{border-left:3px solid #b14dff;background:rgba(177,77,255,.08);padding:.5rem .8rem;border-radius:6px;font-size:.82rem;color:#c9d6f5}
button[data-baseweb="tab"]{font-family:'Orbitron',sans-serif;font-size:.72rem}

.stApp,.stApp p,.stApp li,.stApp label,.stApp span,.stApp [data-testid="stWidgetLabel"] *,.stApp [data-testid="stCaptionContainer"] *,.stApp summary{color:#eaf4ff !important}
.stApp .asz-title{-webkit-text-fill-color:transparent}
.stApp .card .lbl{color:#9fb0d8 !important}.stApp .card .val{color:#00f0ff !important}.stApp .card.buy .val{color:#00e676 !important}.stApp .card.sell .val{color:#ff1744 !important}.stApp .card.wait .val,.stApp .card.warn .val{color:#ffd54f !important}.stApp .card .sub{color:#c3d1f0 !important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:#0f1630 !important;border:1px solid #2a3a7a !important}
div[data-baseweb="select"] *,input{color:#ffffff !important}
ul[role="listbox"],li[role="option"],div[data-baseweb="popover"] *{background:#0f1630 !important;color:#ffffff !important}
li[role="option"]:hover{background:#1c2a66 !important}
button[data-baseweb="tab"]{color:#bcd0ff !important} button[data-baseweb="tab"][aria-selected="true"]{color:#00f0ff !important}
pre,code{background:#0b1024 !important;color:#cfe8ff !important}
details{background:rgba(15,22,48,.7);border:1px solid #243069 !important;border-radius:10px}
.stButton>button{width:100%;background:#14204a;color:#fff !important;border:1px solid #2a3a7a;border-radius:10px}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{background:linear-gradient(90deg,#00c3ff,#7b2ff7);border:0;font-family:'Orbitron',sans-serif;font-size:1.15rem;letter-spacing:.12em;padding:.85rem;box-shadow:0 0 22px rgba(0,240,255,.55);animation:pulse 1.6s infinite}
@keyframes pulse{50%{box-shadow:0 0 38px rgba(177,77,255,.9)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pop{from{opacity:0;transform:translateY(14px) scale(.97)}to{opacity:1;transform:none}}
.card{animation:pop .5s ease-out}
.scan{text-align:center;padding:1.4rem;border:1px solid #243069;border-radius:14px;background:rgba(10,14,30,.85);margin:.6rem 0}
.ring{width:70px;height:70px;margin:0 auto 12px;border-radius:50%;border:4px solid transparent;border-top-color:#00f0ff;border-right-color:#b14dff;animation:spin .9s linear infinite}
.scan-txt{color:#00f0ff;font-family:'Orbitron',sans-serif;letter-spacing:.08em}
.scan-sub{color:#9fb0d8;font-size:.75rem}
.bar{height:8px;background:#14204a;border-radius:6px;overflow:hidden;margin:12px auto;max-width:520px}
.fill{height:100%;background:linear-gradient(90deg,#00f0ff,#b14dff);box-shadow:0 0 12px #00f0ff;transition:width .4s}
.tblwrap{overflow:auto;border:1px solid #243069;border-radius:10px;margin:.4rem 0}
table.asz-tbl{width:100%;border-collapse:collapse;font-size:.82rem}
table.asz-tbl th{background:#14204a;color:#00f0ff !important;text-align:left;padding:6px 10px;position:sticky;top:0}
table.asz-tbl td{color:#eaf4ff !important;padding:5px 10px;border-bottom:1px solid #1b2757;background:rgba(10,14,30,.7)}
</style>
"""


def card(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return f'<div class="card {cls}"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def dec(symbol: str) -> int:
    return 3 if "JPY" in symbol else 5


def pip(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def tick_age_s(t: dict) -> Optional[float]:
    if t.get("quoteAgeSeconds") is not None:
        return float(t["quoteAgeSeconds"])
    try:
        return (pd.Timestamp.now(tz="UTC") - pd.to_datetime(t["lastQuoteAt"], utc=True)).total_seconds()
    except Exception:
        return None


def live_panel(symbol: str) -> None:
    t, err = try_call(fetch_tick, symbol)
    if err:
        st.error(f"Live quote unavailable for {symbol}: {err}")
        return
    st.session_state["last_ok"] = dt.datetime.now(dt.timezone.utc)
    age = tick_age_s(t)
    stale = t["stale"] or (age is not None and age > 60)
    dd = dec(symbol)
    state = t["marketState"].upper()
    c = st.columns(6)
    chg = t["dayDiffPercent"]
    items = [
        ("Mid price", f"{t['mid']:.{dd}f}", f"day {chg:+.2f}%" if (chg is not None and state == "OPEN") else "day change hidden (market not open)", ""),
        ("Bid", f"{t['bid']:.{dd}f}", "", ""),
        ("Ask", f"{t['ask']:.{dd}f}", "", ""),
        ("Spread", f"{t['spread'] / pip(symbol):.1f} pips", f"{t['spread']:.{dd + 1}f}", ""),
        ("Market status", state, "last price only" if state == "CLOSED" else (t["source"] or ""), "warn" if state != "OPEN" else ""),
        ("Quote freshness", "STALE" if stale else "FRESH", f"{age:.0f}s old | {t['lastQuoteAt']}" if age is not None else str(t["lastQuoteAt"]), "warn" if stale else ""),
    ]
    for col, (a, b, s, k) in zip(c, items):
        col.markdown(card(a, b, s, k), unsafe_allow_html=True)


def make_chart(df: pd.DataFrame, n_show: int, title: str):
    d = add_indicators(df).tail(n_show)
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.015, row_heights=[0.44, 0.12, 0.14, 0.15, 0.15])
    x = d["time"]
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Price",
                                 increasing_line_color="#00e676", decreasing_line_color="#ff1744"), 1, 1)
    for col, color in (("ema9", "#00f0ff"), ("ema21", "#b14dff"), ("ema50", "#ffd54f"), ("sma20", "#7f8fb8"), ("sma100", "#ff9100")):
        fig.add_trace(go.Scatter(x=x, y=d[col], name=col.upper(), line=dict(width=1.1, color=color)), 1, 1)
    for col in ("bb_u", "bb_l"):
        fig.add_trace(go.Scatter(x=x, y=d[col], name="BB" if col == "bb_u" else "BB lower", showlegend=col == "bb_u",
                                 line=dict(width=1, dash="dot", color="#4d7cff")), 1, 1)
    vcol = np.where(d["close"] >= d["open"], "#00e676", "#ff1744")
    fig.add_trace(go.Bar(x=x, y=d["tickVolume"], marker_color=vcol, name="Tick volume"), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=d["rsi14"], name="RSI14", line=dict(color="#00f0ff", width=1.2)), 3, 1)
    for lv in (30, 70):
        fig.add_hline(y=lv, line=dict(color="#555", dash="dot", width=1), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=d["macdh"], name="MACD hist", marker_color=np.where(d["macdh"] >= 0, "#00e676", "#ff1744")), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=d["macd"], name="MACD", line=dict(color="#00f0ff", width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=d["macds"], name="Signal", line=dict(color="#b14dff", width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=d["atr14"], name="ATR14", line=dict(color="#ffd54f", width=1.2)), 5, 1)
    fig.update_layout(template="plotly_dark", height=880, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,14,30,0.6)", title=title,
                      legend=dict(orientation="h", y=1.04), dragmode="pan")
    fig.update_xaxes(rangeslider_visible=False, showspikes=True, spikemode="across", spikethickness=1, spikecolor="#7f8fb8")
    fig.update_yaxes(showspikes=True, spikethickness=1, spikecolor="#7f8fb8")
    return fig


def show_df(d: pd.DataFrame, height: int = 0) -> None:
    html = d.to_html(index=False, border=0, na_rep="-", classes="asz-tbl")
    style = f' style="max-height:{height}px"' if height else ""
    st.markdown(f'<div class="tblwrap"{style}>{html}</div>', unsafe_allow_html=True)


def safe(label: str, fn, *args) -> None:
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001 - isolate each section
        log(f"{label} section error: {type(e).__name__}: {e}")
        st.error(f"{label} failed: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def anim_html(msg: str, pct: int) -> str:
    return (f'<div class="scan"><div class="ring"></div><div class="scan-txt">{msg}</div>'
            f'<div class="bar"><div class="fill" style="width:{pct}%"></div></div>'
            f'<div class="scan-sub">ASZ NEURAL ENGINE - {pct}%</div></div>')


def run_analysis(df: pd.DataFrame, ph) -> dict:
    T = len(df)
    if T < MIN_BARS:
        raise ValueError(f"Insufficient candle history: {T} bars (need >= {MIN_BARS}). Try a shorter timeframe.")
    split = WARM + int(0.65 * (T - 1 - WARM))
    ctx: dict = {"split": split, "T": T}
    ph.markdown(anim_html("Scanning thousands of indicator configurations...", 25), unsafe_allow_html=True)
    ctx["lab"], ctx["lab_err"] = try_call(run_lab, df, split, WARM)
    time.sleep(0.4)
    ph.markdown(anim_html("Training AI models on chronological history...", 60), unsafe_allow_html=True)
    ctx["ml"], ctx["ml_err"] = try_call(run_ml, df, split, WARM)
    time.sleep(0.4)
    ph.markdown(anim_html("Blending BUY / SELL probability...", 92), unsafe_allow_html=True)
    time.sleep(0.5)
    ph.empty()
    return ctx


def section_prediction(df: pd.DataFrame, ctx: dict, tf: str, symbol: str, thr: float) -> None:
    lab, ml = ctx.get("lab"), ctx.get("ml")
    if lab is None and (ml is None or not ml.get("ok")):
        st.warning("No prediction available: " + str(ctx.get("lab_err") or ctx.get("ml_err") or (ml or {}).get("msg")))
        return
    p_ind = 0.5 + 0.5 * float(lab["cons"][-1]) if lab else 0.5
    ml_ok = bool(ml and ml.get("ok"))
    p_ml = ml["p_live"] if ml_ok else float("nan")
    p = float(blend(np.array(p_ml), np.array(p_ind)))
    edge = ml_ok and ml["edge"]
    lean = "BUY" if p >= thr else "SELL" if p <= 1 - thr else "WAIT"
    action = lean if (lean == "WAIT" or edge) else f"LEAN {lean}"
    cls = "wait" if lean == "WAIT" else ("buy" if lean == "BUY" else "sell")
    strength = abs(p - 0.5) * 200
    mins = TF_MAP[tf][0]
    nxt = pd.to_datetime(df["time"].iloc[-1]) + pd.Timedelta(minutes=mins)
    trend = "UP" if df["close"].iloc[-1] > ema(df["close"], 50).iloc[-1] else "DOWN"
    c = st.columns(5)
    c[0].markdown(card("BUY estimate", f"{p * 100:.1f}%", "model score, not a win rate", "buy" if lean == "BUY" else ""), unsafe_allow_html=True)
    c[1].markdown(card("SELL estimate", f"{(1 - p) * 100:.1f}%", "model score, not a win rate", "sell" if lean == "SELL" else ""), unsafe_allow_html=True)
    c[2].markdown(card("Next-candle call", action, f"candle opening {nxt:%H:%M} UTC ({tf})", cls), unsafe_allow_html=True)
    c[3].markdown(card("Signal strength", f"{strength:.0f}/100", f"trend vs EMA50: {trend}"), unsafe_allow_html=True)
    status = "NO OOS EDGE" if ml_ok and not edge else ("VALIDATED" if ml_ok else "ML OFF")
    c[4].markdown(card("Validation", status, f"OOS acc {ml['acc'] * 100:.1f}% (n={ml['n_test']})" if ml_ok else str(ml and ml.get("msg") or ctx["ml_err"]),
                       "" if status == "VALIDATED" else "warn"), unsafe_allow_html=True)
    st.markdown(
        f'<div class="note">Prediction time {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC | last closed candle {df["time"].iloc[-1]:%Y-%m-%d %H:%M} UTC. '
        'Percentages are <b>uncalibrated model estimates</b> (blend 60% ML + 40% indicator vote). They are not guaranteed probabilities. '
        '"LEAN" means the ML model has not shown a statistically significant out-of-sample edge over a coin flip.</div>', unsafe_allow_html=True)
    with st.expander("Model explanation"):
        st.write(f"**ML P(up)**: {p_ml:.3f}" if ml_ok else f"ML unavailable: {ml and ml.get('msg') or ctx['ml_err']}")
        st.write(f"**Indicator vote score** (mean of {len(lab['sel']) if lab else 0} selected configurations, mapped to 0-1): {p_ind:.3f}")
        if ml_ok:
            st.write(f"Top ML features: {', '.join(ml['importance'].head(6).index)}")
            st.write(f"Validation: accuracy {ml['acc'] * 100:.2f}% vs majority baseline {ml['base_acc'] * 100:.2f}%, Brier {ml['brier']:.4f} (0.25 = coin flip), "
                     f"log-loss {ml['logloss']:.4f} (0.693 = coin flip), z={ml['z']:.2f}, train n={ml['n_train']}, test n={ml['n_test']}.")
            show_df(ml["calibration"].round(3))
    mtf = []
    for lab_tf in ("5 Minutes", "15 Minutes"):
        d2, e2 = try_call(get_candles, symbol, lab_tf)
        if d2 is not None and len(d2) > 60:
            c2 = d2["close"]
            tr = 1 if ema(c2, 21).iloc[-1] > ema(c2, 50).iloc[-1] and c2.iloc[-1] > ema(c2, 21).iloc[-1] else -1 if ema(c2, 21).iloc[-1] < ema(c2, 50).iloc[-1] and c2.iloc[-1] < ema(c2, 21).iloc[-1] else 0
            mtf.append({"Timeframe": lab_tf, "Trend": {1: "Bullish", -1: "Bearish", 0: "Mixed"}[tr], "RSI14": round(float(rsi(c2, 14).iloc[-1]), 1)})
        else:
            mtf.append({"Timeframe": lab_tf, "Trend": f"unavailable ({e2})", "RSI14": None})
    st.caption("Multi-timeframe confirmation (display only; not part of the probability):")
    show_df(pd.DataFrame(mtf))


def section_technical(df: pd.DataFrame, symbol: str) -> None:
    t, info = tech_table(df)
    px = info["px"]
    sup, res, fib, piv = sr_levels(df, px)
    dd = dec(symbol)
    c = st.columns(4)
    c[0].markdown(card("Trend / momentum votes", f"{info['net_votes']:+d}", f"net bullish minus bearish of {info['n']} readings"), unsafe_allow_html=True)
    c[1].markdown(card("Volatility (ATR14)", f"{info['atr'] / pip(symbol):.1f} pips", f"{info['atr'] / px * 1e4:.1f} bps"), unsafe_allow_html=True)
    c[2].markdown(card("Nearest support", f"{sup:.{dd}f}", "confirmed swing low (display only)"), unsafe_allow_html=True)
    c[3].markdown(card("Nearest resistance", f"{res:.{dd}f}", "confirmed swing high (display only)"), unsafe_allow_html=True)
    show_df(t)
    pats = candle_patterns(df["open"], df["high"], df["low"], df["close"])
    found = [f"{n}: {'bullish' if v[0][-1] else 'bearish'}" for n, v in pats.items() if v[0][-1] or v[1][-1]]
    st.write("**Candlestick analysis (last closed candle):** " + (", ".join(found) if found else "no pattern detected"))
    st.write(f"**Tick volume:** last-bar z-score vs 50 bars = {info['tv_z']:+.2f} -> {'unusually high activity' if info['tv_z'] > 2 else 'unusually low activity' if info['tv_z'] < -2 else 'normal activity'}. "
             "Tick volume counts price updates from the provider; it is not centralized traded volume.")
    st.write("**Rolling-window pivots:** " + ", ".join(f"{k} {v:.{dd}f}" for k, v in piv.items()))
    st.write("**Fibonacci retracements (last 150 bars):** " + ", ".join(f"{k}: {v:.{dd}f}" for k, v in fib.items()))


def section_chart(df: pd.DataFrame, symbol: str, tf: str) -> None:
    n = st.slider("Candles shown", 50, min(len(df), 600), min(200, len(df)), key="nshow")
    if PLOTLY_OK:
        st.plotly_chart(make_chart(df, n, f"{symbol} - {tf}"), config={"scrollZoom": True, "displaylogo": False})
    else:
        st.warning(f"Plotly unavailable ({PLOTLY_ERR}); showing a basic close-price line chart instead.")
        st.line_chart(df.set_index("time")["close"].tail(n))


def section_lab(ctx: dict) -> None:
    lab = ctx.get("lab")
    if lab is None:
        st.warning(f"Indicator lab unavailable: {ctx.get('lab_err')}")
        return
    c = st.columns(4)
    c[0].markdown(card("Configurations", f"{lab['n_cfg']:,}", f"parameter variants across {lab['n_fam']} families, NOT independent indicators"), unsafe_allow_html=True)
    c[1].markdown(card("Eligible", f"{lab['n_eligible']:,}", f"need >= {lab['min_sig']} in-sample signals"), unsafe_allow_html=True)
    c[2].markdown(card("Selected for engine", f"{len(lab['sel'])}", "top by in-sample Wilson lower bound"), unsafe_allow_html=True)
    c[3].markdown(card("Selected OOS hit", f"{lab['sel_oos_hit']:.1f}%", f"on {lab['sel_oos_n']} held-out signals"), unsafe_allow_html=True)
    st.markdown(f'<div class="note">Ranking uses only the first {lab["split"] - lab["warm"]} usable bars (in-sample). The final {ctx["T"] - 1 - lab["split"]} bars are held out and never used for selection. '
                f'With thousands of configurations, the best in-sample results are partly luck (multiple-comparison bias): only {lab["pct_oos_above_50"]:.1f}% of eligible configurations beat 50% out-of-sample. '
                f'Computed in {lab["elapsed"]:.1f}s and cached.</div>', unsafe_allow_html=True)
    st.subheader("Top 150 by in-sample rank (with out-of-sample result)")
    show_df(lab["top"].round(2), 480)
    st.subheader("Configurations used by the prediction engine")
    st.code("\n".join(lab["sel_names"]))
    st.subheader("Family comparison")
    show_df(lab["families"].round(2))


def section_backtest(df: pd.DataFrame, ctx: dict, thr: float, spread: float, symbol: str) -> None:
    lab, ml = ctx.get("lab"), ctx.get("ml")
    if lab is None:
        st.warning("Backtest needs the indicator lab: " + str(ctx.get("lab_err")))
        return
    o, c = df["open"].values.astype(float), df["close"].values.astype(float)
    T, split = len(df), ctx["split"]
    p_ind = 0.5 + 0.5 * lab["cons"]
    p_ml = ml["p_test"] if ml and ml.get("ok") else np.full(T, np.nan)
    strategies = {"Indicators only": p_ind, "Combined (ML + indicators)": blend(p_ml, p_ind)}
    if ml and ml.get("ok"):
        strategies["ML only"] = p_ml
    res, rows = {}, []
    for name, p in strategies.items():
        r = backtest(o, c, p, split, T - 1, thr, spread)
        res[name] = r
        rows.append({"Strategy": name, "Test bars": r["bars"], "Signals": r["signals"], "No-trade": r["no_trade"], "BUY": r["buys"], "SELL": r["sells"],
                     "Directional hit % (gross)": r["hit_rate"], "Net win % (after spread)": r["net_win_rate"], "Wins": r["wins"], "Losses": r["losses"],
                     "Profit factor": r["profit_factor"], "Total return (bps)": r["total_bps"], "Max drawdown (bps)": r["max_dd_bps"]})
    show_df(pd.DataFrame(rows).round(2))
    best = res.get("Combined (ML + indicators)")
    if best is not None and len(best["eq"]):
        eq = pd.DataFrame({"Equity (bps, non-compounded)": best["eq"]}, index=df["time"].iloc[best["t"] + 1].values)
        if PLOTLY_OK:
            fig = go.Figure(go.Scatter(x=eq.index, y=eq.iloc[:, 0], line=dict(color="#00f0ff"), name="Equity"))
            fig.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)", title="Combined strategy equity curve (held-out period)", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig)
        else:
            st.line_chart(eq)
    st.markdown(
        f'<div class="note"><b>Three different things:</b> (1) directional hit rate = how often the next candle went the predicted way; (2) strategy profitability = P&L after costs '
        f'(entry at next open, exit at that candle close, cost = latest quoted spread of {spread / pip(symbol):.1f} pips on every trade, which ignores spread widening, slippage and swaps); '
        f'(3) live performance = unknown until you track it. Only the held-out period is shown; the ML model was trained on earlier data only. '
        'Provider candles may be bid- or mid-based (not documented), so real fills can differ.</div>', unsafe_allow_html=True)


def section_watchlist(symbols: List[str]) -> None:
    default = [s for s in MAJORS if s in symbols][:8] or symbols[:8]
    pick = st.multiselect("Watchlist (type to search)", symbols, default=default, key="wl")
    if not pick:
        st.info("Select at least one pair.")
        return
    ticks = fetch_latest(tuple(pick))
    rows = []
    for s in pick:
        t = ticks.get(s)
        if not t:
            rows.append({"Pair": s, "Status": "no quote returned"})
            continue
        age = tick_age_s(t)
        rows.append({"Pair": s, "Bid": t["bid"], "Ask": t["ask"], "Mid": t["mid"], "Spread (pips)": (t["spread"] or 0) / pip(s),
                     "Day %": t["dayDiffPercent"] if t["marketState"] == "open" else None, "Market": t["marketState"],
                     "Age (s)": age, "Fresh": "no" if (t["stale"] or (age or 0) > 60) else "yes"})
    show_df(pd.DataFrame(rows))


def section_system(df: Optional[pd.DataFrame], ctx: Optional[dict], sym_err: Optional[str], df_err: Optional[str]) -> None:
    import importlib.metadata as md

    def ver(p: str) -> str:
        try:
            return md.version(p)
        except Exception:
            return "NOT INSTALLED"
    h, he = try_call(lambda: _get("/health"))
    c = st.columns(4)
    c[0].markdown(card("Provider", "ONLINE" if h is not None else "ERROR", he or "biquote.io /health reachable", "" if h is not None else "warn"), unsafe_allow_html=True)
    lo = st.session_state.get("last_ok")
    c[1].markdown(card("Last live quote", f"{lo:%H:%M:%S} UTC" if lo else "none", "last successful tick fetch"), unsafe_allow_html=True)
    ml = (ctx or {}).get("ml")
    c[2].markdown(card("ML model", "READY" if ml and ml.get("ok") else "OFF", (ml or {}).get("msg", (ctx or {}).get("ml_err", "")) or ""), unsafe_allow_html=True)
    c[3].markdown(card("Candles", f"{len(df) if df is not None else 0}", df_err or (f"last {df['time'].iloc[-1]:%Y-%m-%d %H:%M} UTC" if df is not None and len(df) else "")), unsafe_allow_html=True)
    show_df(pd.DataFrame({"Package": p, "Version": ver(p)} for p in ("streamlit", "pandas", "numpy", "requests", "plotly", "scikit-learn")))
    if sym_err:
        st.warning(f"Symbol discovery failed ({sym_err}); using built-in major-pair list. Quotes are still fetched live.")
    if h is not None:
        st.code(json.dumps(h, indent=2, default=str))
    st.subheader("Error diagnostics")
    st.code("\n".join(st.session_state.get("diag", [])) or "no errors logged")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="ASZ AI FOREX TERMINAL", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("diag", [])

    syms, sym_err = try_call(fetch_symbols)
    names = [s[0] for s in syms] if syms else MAJORS
    desc = dict(syms) if syms else {}

    with st.sidebar:
        st.markdown('<p class="asz-title">ASZ</p><div class="asz-sub">AI FOREX TERMINAL</div>', unsafe_allow_html=True)
        symbol = st.selectbox("Currency pair (type to search)", names, index=names.index("EURUSD") if "EURUSD" in names else 0,
                              format_func=lambda s: f"{s} - {desc[s]}" if desc.get(s) else s)
        tf = st.selectbox("Prediction timeframe", list(TF_MAP), index=3)
        auto = st.toggle("Auto-refresh live quote", value=True)
        every = st.select_slider("Refresh every (s)", [5, 10, 15, 30, 60], value=10)
        thr = st.slider("BUY/SELL threshold", 0.51, 0.70, 0.55, 0.01, help="Combined score >= threshold -> BUY, <= 1-threshold -> SELL, else WAIT")
        if st.button("Manual refresh (clear cache)"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Data: biquote.io public API (no key). Forex only. Tick volume = provider tick count, not centralized volume.")

    st.markdown('<p class="asz-title">ASZ AI FOREX TERMINAL</p><div class="asz-sub">QUANTITATIVE FOREX ANALYSIS - MODEL ESTIMATES, NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)

    st.subheader("A. Live market")
    if auto and hasattr(st, "fragment"):
        st.fragment(run_every=f"{every}s")(live_panel)(symbol)
    else:
        live_panel(symbol)

    df, df_err = try_call(get_candles, symbol, tf)
    if df is not None and len(df) < 60:
        df_err, df = f"Only {len(df)} complete {tf} candles available", None
    tk, _ = try_call(fetch_tick, symbol)
    spread = float(tk["spread"]) if tk and tk.get("spread") else 0.0
    if df is None:
        st.error(f"Candle data unavailable: {df_err}")

    st.subheader("B. AI prediction")
    clicked = st.button("⚡ ANALYZE", type="primary", key="analyze")
    ph = st.empty()
    if clicked:
        if df is None:
            st.error("Cannot analyze: no candle data.")
        else:
            try:
                ctx_new = run_analysis(df, ph)
                st.session_state["res"] = {"symbol": symbol, "tf": tf, "df": df, "ctx": ctx_new}
            except Exception as e:  # noqa: BLE001
                ph.empty()
                log(f"analysis error: {e}")
                st.error(f"Analysis failed: {e}")
    res = st.session_state.get("res")
    ok = bool(res and res["symbol"] == symbol and res["tf"] == tf)
    adf, ctx = (res["df"], res["ctx"]) if ok else (None, None)
    if ok:
        safe("AI prediction", section_prediction, adf, ctx, tf, symbol, thr)
    elif res:
        st.info("Pair or timeframe changed - press ANALYZE for a fresh BUY/SELL estimate.")
    else:
        st.info("Press ANALYZE to scan indicators, train the models and see the BUY / SELL probability.")

    tabs = st.tabs(["C. Technical", "Chart", "D. Indicator Lab", "E. Backtesting", "F. Watchlist", "G. System"])
    with tabs[0]:
        if df is not None:
            safe("Technical analysis", section_technical, df, symbol)
    with tabs[1]:
        if df is not None:
            safe("Chart", section_chart, df, symbol, tf)
    with tabs[2]:
        if ok:
            safe("Indicator lab", section_lab, ctx)
        else:
            st.info("Press ANALYZE first.")
    with tabs[3]:
        if ok:
            safe("Backtesting", section_backtest, adf, ctx, thr, spread, symbol)
        else:
            st.info("Press ANALYZE first.")
    with tabs[4]:
        safe("Watchlist", section_watchlist, names)
    with tabs[5]:
        safe("System status", section_system, df, ctx, sym_err, df_err)


if __name__ == "__main__":
    main()
