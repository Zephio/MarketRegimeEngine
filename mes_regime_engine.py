"""
MES Regime Engine
=================

This module implements a full market regime detection engine for the
Micro E‑mini S&P 500 futures (MES) using only OHLCV data.  It can be
run as a standalone script to connect to Interactive Brokers (IBKR)
via the `ib_insync` API, collect historical bars, detect change
points, train a Hidden Markov Model (HMM) with three regimes, and
stream regime probabilities in real time.

The code is organised into several sections:

- **RegimeConfig** defines all tunable parameters.
- **Feature engineering** computes statistical features from OHLCV data.
- **Change point detection** uses `ruptures` to locate structural
  breaks.
- **RegimeHMM** wraps an HMM from `hmmlearn`, automatically mapping
  states to labels (trend, mean‑reversion, chop).
- **IBKR helpers** fetch historical bars, pick the front‑month MES
  contract, and place example orders. Trading is disabled by default.
- **Main loop** ties everything together: connect, train, stream,
  compute probabilities, and decide the current trading mode.

See the `README.md` for usage instructions and important caveats.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt

from ib_insync import IB, Future, MarketOrder, util  # type: ignore


###############################################################################
# Configuration
###############################################################################


@dataclass
class RegimeConfig:
    """Configuration parameters for the regime engine."""

    # Data collection
    bar_size: str = "15 mins"       # IBKR bar size (e.g., "5 mins", "30 mins")
    duration: str = "30 D"          # Historical lookback duration (e.g., "10 D", "1 M")
    use_rth: bool = False            # Use only regular trading hours
    keep_up_to_date: bool = True     # Stream bars in real time

    # Feature lookbacks
    vol_lookback: int = 96           # Bars for volatility & ATR
    trend_lookback: int = 96         # Bars for EMA & autocorrelation
    bb_lookback: int = 96            # Bars for Bollinger band z‑score
    volume_lookback: int = 96        # Bars for volume surprise

    # Change point detection
    cp_window: int = 1000            # Bars to examine for change points
    cp_penalty: float = 10.0         # PELT penalty term (larger = fewer CPs)
    min_bars_since_cp: int = 24      # Bars to wait after a CP before acting

    # HMM configuration
    n_states: int = 3                # Trend, mean‑revert, chop
    hmm_train_window: int = 3000     # Training window for HMM
    hmm_retrain_every: int = 48      # Retrain every N bars (~ once per day)
    hmm_cov_type: str = "full"      # Covariance type: "diag" or "full"
    hmm_iter: int = 200              # EM iterations
    hmm_seed: int = 7                # Random seed

    # Gating thresholds
    p_enter: float = 0.70            # Enter regime if probability ≥ this
    p_exit: float = 0.55             # Exit regime if probability < this

    # Trading parameters (disabled by default)
    trade_enabled: bool = False      # Enable trading logic
    paper_account_only: bool = True  # Only run on paper account
    max_position: int = 1            # Example target position (contracts)

    # MES contract & exchange
    exchange: str = "CME"           # Exchange for MES
    currency: str = "USD"           # Currency


###############################################################################
# Feature engineering
###############################################################################


def _safe_log_return(close: pd.Series) -> pd.Series:
    """Compute log returns with NaN for non‑finite values."""
    prev = close.shift(1)
    lr = np.log(close / prev)
    return lr.replace([np.inf, -np.inf], np.nan)


def compute_features(df: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    """
    Compute a set of time series features from OHLCV data. The input
    DataFrame must have columns `open`, `high`, `low`, `close`, and
    `volume`. The returned DataFrame contains the original series as
    well as derived features used for HMM modelling.
    """
    out = df.copy()

    # Log returns
    out["lr"] = _safe_log_return(out["close"])

    # Realised volatility (rolling standard deviation)
    out["rv"] = out["lr"].rolling(cfg.vol_lookback).std()

    # Exponential moving average and fractional slope
    ema = out["close"].ewm(span=cfg.trend_lookback, adjust=False).mean()
    out["ema"] = ema
    out["ema_slope"] = (ema - ema.shift(1)) / ema.shift(1)

    # Bollinger z‑score
    ma = out["close"].rolling(cfg.bb_lookback).mean()
    sd = out["close"].rolling(cfg.bb_lookback).std()
    out["bb_z"] = (out["close"] - ma) / sd.replace(0, np.nan)

    # Volume surprise (z‑score)
    vma = out["volume"].rolling(cfg.volume_lookback).mean()
    vsd = out["volume"].rolling(cfg.volume_lookback).std()
    out["vol_z"] = (out["volume"] - vma) / vsd.replace(0, np.nan)

    # Short‑term autocorrelation of returns (lag‑1)
    out["ac1"] = out["lr"].rolling(cfg.trend_lookback).corr(out["lr"].shift(1))

    # Average true range percentage (proxy for market chop)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(cfg.vol_lookback).mean()
    out["atr_pct"] = out["atr"] / out["close"]

    # Combine features; to make sure all feature columns exist
    features = ["lr", "rv", "ema_slope", "bb_z", "vol_z", "ac1", "atr_pct"]
    return out[features + ["open", "high", "low", "close", "volume"]]


###############################################################################
# Change point detection
###############################################################################


def detect_change_point_index(feat_df: pd.DataFrame, cfg: RegimeConfig) -> Optional[int]:
    """
    Detect the most recent change point in the multivariate feature set
    using the PELT algorithm from the `ruptures` library. Returns the
    integer position of the last change point in `feat_df` or `None`
    if no change point is detected.
    """
    # Choose a subset of features for CP detection
    use_cols = ["lr", "rv", "ema_slope", "atr_pct", "vol_z"]
    X = feat_df[use_cols].dropna()

    # Need sufficient history to detect CP
    min_history = max(200, cfg.min_bars_since_cp + 5)
    if len(X) < min_history:
        return None

    # Use only the last `cp_window` bars
    windowed = X.iloc[-cfg.cp_window:] if len(X) > cfg.cp_window else X
    data = windowed.values

    # Apply PELT change point detection with RBF kernel
    algo = rpt.Pelt(model="rbf").fit(data)
    bkps = algo.predict(pen=cfg.cp_penalty)
    # Remove the last breakpoint (end of series)
    cps = [b for b in bkps if b < len(data)]
    if not cps:
        return None

    last_cp_local = cps[-1]

    # Map local index to global position
    global_indices = feat_df.index.get_indexer(windowed.index)
    # `last_cp_local` is one‐based in ruptures output; subtract 1 for 0‑based index
    global_pos = global_indices[last_cp_local - 1]
    return int(global_pos)


###############################################################################
# Hidden Markov Model regime classifier
###############################################################################


class RegimeHMM:
    """
    Wrapper around a Gaussian HMM that learns latent regime states
    and maps them to semantic labels (trend, mean‑revert, chop).
    """

    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model: Optional[GaussianHMM] = None
        self.state_map: Optional[Dict[int, str]] = None
        self.bars_seen = 0

    def _prepare(self, feat_df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        cols = ["lr", "rv", "ema_slope", "bb_z", "vol_z", "ac1", "atr_pct"]
        x = feat_df[cols].dropna()
        return x.values, x.index

    def _fit(self, feat_df: pd.DataFrame) -> None:
        X, idx = self._prepare(feat_df)
        if len(X) < 400:
            return
        # Use recent training window
        X_sub = X[-self.cfg.hmm_train_window:] if len(X) > self.cfg.hmm_train_window else X
        X_scaled = self.scaler.fit_transform(X_sub)
        hmm = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.hmm_cov_type,
            n_iter=self.cfg.hmm_iter,
            random_state=self.cfg.hmm_seed,
        )
        hmm.fit(X_scaled)
        self.model = hmm
        self.state_map = self._map_states(X_sub, hmm)

    def _map_states(self, X_unscaled: np.ndarray, hmm: GaussianHMM) -> Dict[int, str]:
        """
        Inspect the posterior‑weighted means of each state to assign
        human‑interpretable labels. A simple heuristic is used:

        - **Trend**: large absolute EMA slope, positive autocorr, moderate return
        - **Mean‑revert**: negative autocorr and large Bollinger z
        - **Chop**: high ATR/volatility but low slope
        """
        X_scaled = self.scaler.transform(X_unscaled)
        post = hmm.predict_proba(X_scaled)  # shape (T, K)
        # compute weighted average of unscaled features per state
        feats_names = ["lr", "rv", "ema_slope", "bb_z", "vol_z", "ac1", "atr_pct"]
        state_stats: Dict[int, Dict[str, float]] = {}
        for k in range(hmm.n_components):
            weights = post[:, k]
            w_sum = weights.sum() + 1e-12
            mu_scaled = (X_scaled * weights[:, None]).sum(axis=0) / w_sum
            mu_unscaled = self.scaler.inverse_transform(mu_scaled.reshape(1, -1)).ravel()
            state_stats[k] = dict(zip(feats_names, mu_unscaled))

        # Compute scores for each label and assign states greedily
        scores: Dict[int, Dict[str, float]] = {}
        for k, s in state_stats.items():
            trend_score = (abs(s["ema_slope"]) * 3) + (max(s["ac1"], 0) * 2) + (abs(s["lr"]) * 0.5)
            mr_score = (max(-s["ac1"], 0) * 2.5) + (abs(s["bb_z"]) * 1.5)
            chop_score = (s["atr_pct"] * 3) + (s["rv"] * 2) - (abs(s["ema_slope"]) * 1)
            scores[k] = {
                "trend": trend_score,
                "mean_revert": mr_score,
                "chop": chop_score,
            }

        # Greedy assignment: pick the best state for each label
        mapping: Dict[int, str] = {}
        remaining = set(range(hmm.n_components))
        def assign(label: str):
            best = max(remaining, key=lambda k: scores[k][label])
            mapping[best] = label
            remaining.remove(best)

        assign("trend")
        if remaining:
            assign("mean_revert")
        # Remaining state(s) → chop
        for k in remaining:
            mapping[k] = "chop"
        return mapping

    def maybe_retrain(self, feat_df: pd.DataFrame) -> None:
        """Retrain the HMM if the configured period has elapsed."""
        self.bars_seen += 1
        if self.model is None:
            self._fit(feat_df)
        elif self.bars_seen % self.cfg.hmm_retrain_every == 0:
            self._fit(feat_df)

    def regime_probs(self, feat_df: pd.DataFrame) -> Optional[Dict[str, float]]:
        if self.model is None or self.state_map is None:
            return None
        X, idx = self._prepare(feat_df)
        if len(X) < 20:
            return None
        x_last = X[-1:]
        x_scaled = self.scaler.transform(x_last)
        post = self.model.predict_proba(x_scaled)[0]
        out: Dict[str, float] = {"trend": 0.0, "mean_revert": 0.0, "chop": 0.0}
        for k, p in enumerate(post):
            label = self.state_map.get(k, "chop")
            out[label] += float(p)
        # Normalise
        total = sum(out.values())
        if total > 0:
            for k in out:
                out[k] /= total
        return out


###############################################################################
# Interactive Brokers helpers
###############################################################################


def get_front_month_mes(ib: IB, exchange: str = "CME") -> Future:
    """
    Determine the current front‑month Micro E‑mini S&P 500 futures contract.

    IBKR returns a list of contract details for a generic MES definition. We
    filter those with a valid `lastTradeDateOrContractMonth`, sort by
    expiry, and pick the nearest unexpired contract. You can refine
    this logic by checking volume or open interest via market data.
    """
    generic = Future(symbol="MES", exchange=exchange, currency="USD")
    cds = ib.reqContractDetails(generic)
    if not cds:
        raise RuntimeError("No contract details returned for MES. Check market data permissions.")
    # Filter futures with valid lastTradeDateOrContractMonth
    futs = [cd.contract for cd in cds if getattr(cd.contract, "lastTradeDateOrContractMonth", None)]
    if not futs:
        raise RuntimeError("No valid MES future found.")
    # Sort by contract month: treat both YYYYMM and YYYYMMDD consistently
    def expiry_key(c: Future) -> int:
        s = c.lastTradeDateOrContractMonth
        # Use the first 8 digits (YYYYMMDD) if available; else fallback to YYYYMM*100 + 1
        if len(s) >= 8:
            return int(s[:8])
        return int(s[:6]) * 100 + 1
    futs_sorted = sorted(futs, key=expiry_key)
    front = futs_sorted[0]
    # Qualify contract to set conId and localSymbol
    ib.qualifyContracts(front)
    return front


def ib_bars_to_df(bars) -> pd.DataFrame:
    """Convert IBKR bar data to a pandas DataFrame indexed by datetime."""
    df = util.df(bars)
    df = df.rename(columns={"date": "datetime"})
    df = df.set_index("datetime").sort_index()
    # Ensure columns exist
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    if "volume" not in df.columns:
        df["volume"] = np.nan
    else:
        df["volume"] = df["volume"].astype(float)
    return df


def place_order_example(ib: IB, contract: Future, target_pos: int, cfg: RegimeConfig) -> None:
    """
    Illustrative order function. Converts a target position into a market
    order. This function is disabled unless `cfg.trade_enabled` is set
    to True. In production you must implement proper position sizing,
    risk checks, and account reconciliation.
    """
    if not cfg.trade_enabled:
        return
    # Determine whether to buy or sell
    if target_pos == 0:
        return
    action = "BUY" if target_pos > 0 else "SELL"
    qty = abs(int(target_pos))
    order = MarketOrder(action, qty)
    trade = ib.placeOrder(contract, order)
    print(f"[ORDER] {action} {qty} {contract.localSymbol}")


###############################################################################
# Decision logic
###############################################################################


def decide_mode(probs: Dict[str, float], last_mode: str, bars_since_cp: int, cfg: RegimeConfig) -> str:
    """
    Determine the current trading mode (trend, mean_revert, chop)
    based on regime probabilities, hysteresis thresholds, and
    change‑point stability.
    """
    # Do not act for a period after a change point
    if bars_since_cp < cfg.min_bars_since_cp:
        return "chop"
    pt, pm = probs.get("trend", 0.0), probs.get("mean_revert", 0.0)
    # Stay in trend until probability drops below exit threshold
    if last_mode == "trend":
        if pt < cfg.p_exit:
            return "chop"
        return "trend"
    # Stay in MR until it drops
    if last_mode == "mean_revert":
        if pm < cfg.p_exit:
            return "chop"
        return "mean_revert"
    # Neutral: check if any regime probability crosses enter threshold
    if pt >= cfg.p_enter and pt >= pm:
        return "trend"
    if pm >= cfg.p_enter and pm > pt:
        return "mean_revert"
    return "chop"


###############################################################################
# Main entry point
###############################################################################


def run(cfg: RegimeConfig) -> None:
    """Run the regime engine using the provided configuration."""
    ib = IB()
    # Connect to IBKR – adjust ports if necessary (paper: 7497, live: 7496)
    try:
        ib.connect("127.0.0.1", 7497, clientId=12)
    except Exception as e:
        raise RuntimeError(f"Could not connect to IBKR: {e}")
    # Determine the current front month contract
    contract = get_front_month_mes(ib, exchange=cfg.exchange)
    print(f"Using contract {contract.localSymbol} (expiry {contract.lastTradeDateOrContractMonth})")
    # Request historical bars with streaming updates
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=cfg.duration,
        barSizeSetting=cfg.bar_size,
        whatToShow="TRADES",
        useRTH=cfg.use_rth,
        formatDate=1,
        keepUpToDate=cfg.keep_up_to_date,
    )
    # Prepare initial data
    df = ib_bars_to_df(bars)
    feat = compute_features(df, cfg)
    hmm = RegimeHMM(cfg)
    hmm.maybe_retrain(feat)
    last_mode = "chop"
    last_cp_pos: Optional[int] = None
    print("Regime engine started. Press Ctrl+C to stop.")
    try:
        while True:
            # Convert new bars to DataFrame
            df = ib_bars_to_df(bars)
            feat = compute_features(df, cfg)
            # Change point detection
            cp_pos = detect_change_point_index(feat, cfg)
            if cp_pos is not None:
                last_cp_pos = cp_pos
            # Bars since last CP
            if last_cp_pos is None:
                bars_since_cp = 10 ** 6
            else:
                bars_since_cp = len(feat) - 1 - last_cp_pos
            # Retrain periodically
            hmm.maybe_retrain(feat)
            probs = hmm.regime_probs(feat)
            if probs is None:
                time.sleep(1)
                continue
            mode = decide_mode(probs, last_mode, bars_since_cp, cfg)
            ts = feat.index[-1]
            close = float(feat["close"].iloc[-1])
            print(
                f"{ts} | close={close:.2f} | P(trend)={probs['trend']:.2f} P(MR)={probs['mean_revert']:.2f} P(chop)={probs['chop']:.2f} | mode={mode} | bars_since_cp={bars_since_cp}"
            )
            # Example order placement (disabled by default)
            if cfg.trade_enabled:
                if mode == "trend":
                    target = cfg.max_position
                elif mode == "mean_revert":
                    target = -cfg.max_position
                else:
                    target = 0
                place_order_example(ib, contract, target, cfg)
            last_mode = mode
            # Sleep a bit to reduce CPU usage
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    # Create a config with default values; users can modify here
    config = RegimeConfig()
    run(config)