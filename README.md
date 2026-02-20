# MES Regime Engine

This repository contains a Python implementation of a **market regime detection engine** for the Micro E‑mini S&P 500 futures (MES) using only OHLCV data. The engine classifies each new bar as belonging to one of three latent regimes—**trend**, **mean‑reversion**, or **chop**—and includes a change‑point detector to avoid acting immediately after regime shifts.

The code is designed to run against Interactive Brokers (IBKR) data using the `ib_insync` library. By default, it automatically selects the current front‑month MES contract and streams 15‑minute bars. You can adjust the bar size and model parameters in the configuration.

## Quick start

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your‑username>/mes_regime_engine.git
   cd mes_regime_engine
   ```

2. **Install dependencies** (ideally in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Interactive Brokers Gateway or Trader Workstation** and ensure it is listening on the default paper trading port (`7497`).

4. **Run the regime engine**:

   ```bash
   python mes_regime_engine.py
   ```

   The script will connect to IBKR, fetch historical data, train a Hidden Markov Model (HMM), and then stream new bars. It prints the regime probabilities and the chosen trading mode (`trend`, `mean_revert`, or `chop`) on each update.

5. **Experiment with parameters**: The `RegimeConfig` dataclass defines lookback windows, change‑point penalties, gating thresholds, and trading parameters. Modify these to suit your timeframe or market.

## Important notes

- **No trading by default**. The sample order placement function is disabled (`trade_enabled=False`) to avoid accidental trades. Set `trade_enabled=True` **only after thorough backtesting** and ensure you understand the risk.
- **Continuous futures**. IBKR does not provide a continuous MES contract. The helper function `get_front_month_mes()` selects the nearest unexpired contract. The script checks once per day for a new front month. For production use, consider a more sophisticated roll rule based on volume or open interest.
- **Dependencies**. Installing `hmmlearn` can be system dependent. If you encounter installation issues, see the `hmmlearn` documentation or install via conda.

## License

This project is provided for educational purposes. Use at your own risk. No warranty is given or implied.