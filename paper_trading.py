import pandas as pd
import numpy as np
import ccxt
import os
import time
from datetime import datetime

print("="*60)
print(" 🚀 PAPER TRADING TRACKER - Versione Definitiva")
print(" Parametri: VOL_H=1.4 | Costi 0.10% | Ribilanciamento 3D")
print("="*60)

# ==========================================
# CONFIGURAZIONE DEFINITIVA
# (versione Originale ottimizzata — migliore Sharpe/DD)
# ==========================================
ASSETS      = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
ASSET_NAMES = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']

Z_THRESH     = 2.0
VOL_H        = 1.4   # soglia Expansion (testato: ottimale)
VOL_L        = 0.7
DD_THRESHOLD = -0.15
MR_Z_THRESH  = -2.5

ALLOCAZIONE = {
    'EXPANSION':   {'tsmom': 0.60, 'mr': 0.00, 'funding': 0.40},
    'NEUTRAL':     {'tsmom': 0.45, 'mr': 0.10, 'funding': 0.45},
    'CONTRACTION': {'tsmom': 0.00, 'mr': 0.10, 'funding': 0.50},
}

# ==========================================
# DATI HOURLY — Regime Detector
# ==========================================
def fetch_hourly_data(symbol):
    exchange = ccxt.kraken({'enableRateLimit': True})
    all_bars = []
    since = exchange.milliseconds() - (1000 * 3600 * 1000)
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe='1h', since=since, limit=720)
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + 3600000
        if len(all_bars) >= 1000:
            break
        time.sleep(0.5)
    df = pd.DataFrame(all_bars, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df['close'].tail(1000)

# ==========================================
# DATI DAILY — TSMOM, Drawdown, MR
# ==========================================
def fetch_daily_data():
    print("📥 Download dati daily da Kraken...")
    exchange = ccxt.kraken({'enableRateLimit': True})
    prices = {}
    for symbol in ASSETS:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=720)
            df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.normalize()
            df.set_index('timestamp', inplace=True)
            prices[symbol.replace('/','')]= df['close']
            print(f"   ✅ {symbol}: {len(df)} giorni")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
    if not prices:
        raise Exception("Nessun dato scaricato")
    close_daily = pd.DataFrame(prices).dropna()
    print(f"   ✅ Totale: {len(close_daily)} giorni")
    return close_daily

# ==========================================
# CALCOLO REGIME + PESI
# ==========================================
def calculate_weights(close_daily, close_hourly_dict):

    # Regime Detector (hourly, moda tra asset)
    print("📊 Calcolo Regime Detector...")
    regime_series_list = []
    for symbol in ASSETS:
        key = symbol.replace('/','')
        if key not in close_hourly_dict:
            continue
        ret_h     = close_hourly_dict[key].pct_change(fill_method=None)
        vol_short = ret_h.rolling(24).std()
        vol_long  = ret_h.rolling(168).std()
        vol_ratio = vol_short / vol_long
        momentum  = ret_h.rolling(24).sum()
        mom_mean  = momentum.rolling(720).mean()
        mom_std   = momentum.rolling(720).std().replace(0, np.nan)
        z_score   = (momentum - mom_mean) / mom_std
        reg_h = pd.Series('NEUTRAL', index=ret_h.index)
        reg_h.loc[(vol_ratio > VOL_H) & (z_score.abs() > Z_THRESH)] = 'EXPANSION'
        reg_h.loc[(vol_ratio < VOL_L) & (z_score.abs() < Z_THRESH)] = 'CONTRACTION'
        regime_series_list.append(
            reg_h.resample('D').agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'NEUTRAL')
        )

    if regime_series_list:
        regime_df  = pd.DataFrame(regime_series_list).T
        regime = regime_df.iloc[-1].mode().iloc[0]
    else:
        regime = 'NEUTRAL'
    print(f"   ✅ Regime: {regime}")

    # TSMOM (segnale a 30 giorni)
    returns_30d = close_daily.pct_change(30, fill_method=None)
    raw_signal  = (returns_30d.iloc[-1] > 0).astype(int)

    # Drawdown Overlay (90 giorni)
    norm_close   = close_daily / close_daily.apply(lambda x: x.dropna().iloc[0])
    market_index = norm_close.mean(axis=1, skipna=True)
    rolling_max  = market_index.rolling(90, min_periods=1).max()
    drawdown     = (market_index.iloc[-1] / rolling_max.iloc[-1]) - 1
    overlay      = 0.5 if drawdown < DD_THRESHOLD else 1.0

    # Mean Reversion
    mr_zscore = (
        (close_daily.iloc[-1] - close_daily.rolling(14).mean().iloc[-1])
        / close_daily.rolling(14).std().iloc[-1]
    )
    sma_50    = close_daily.rolling(50).mean().iloc[-1]
    mr_signal = ((mr_zscore < MR_Z_THRESH) & (close_daily.iloc[-1] > sma_50)).astype(int)

    # Position sizing
    alloc   = ALLOCAZIONE[regime]
    weights = {a: 0.0 for a in ASSET_NAMES}

    if raw_signal.sum() > 0:
        tsmom_w = (raw_signal / raw_signal.sum()) * alloc['tsmom'] * overlay
        for i, a in enumerate(ASSET_NAMES):
            weights[a] += tsmom_w.iloc[i]

    if mr_signal.sum() > 0:
        mr_w = (mr_signal / mr_signal.sum()) * alloc['mr']
        for i, a in enumerate(ASSET_NAMES):
            weights[a] += mr_w.iloc[i]

    total_exp      = sum(weights.values())
    weights['CASH']= max(0.0, 1.0 - total_exp - alloc['funding'])

    return weights, regime, drawdown

# ==========================================
# SALVA RISULTATI
# ==========================================
def save_results(weights, regime, drawdown):
    today = datetime.now().strftime('%Y-%m-%d')
    row   = {
        'date':    today,
        'regime':  regime,
        'drawdown':round(drawdown, 4),
        **{a: round(weights[a], 4) for a in ASSET_NAMES + ['CASH']}
    }

    csv_file = 'paper_trading_log.csv'
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_existing = df_existing[df_existing['date'] != today]
        df_out = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_out = pd.DataFrame([row])

    df_out.to_csv(csv_file, index=False)

    print(f"\n📊 RISULTATI {today}:")
    print(f"   Regime:   {regime}")
    print(f"   Drawdown: {drawdown:.2%}")
    print(f"   Pesi:")
    for a in ASSET_NAMES + ['CASH']:
        print(f"     {a}: {weights[a]:.1%}")
    print(f"   ✅ Salvato in {csv_file}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        close_daily = fetch_daily_data()

        print("📥 Download dati hourly...")
        close_hourly_dict = {}
        for symbol in ASSETS:
            try:
                close_hourly_dict[symbol.replace('/','')]= fetch_hourly_data(symbol)
                print(f"   ✅ Hourly {symbol} OK")
            except Exception as e:
                print(f"   ❌ Hourly {symbol}: {e}")

        weights, regime, drawdown = calculate_weights(close_daily, close_hourly_dict)
        save_results(weights, regime, drawdown)
        print("\n✅ Paper trading completato!")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        raise
