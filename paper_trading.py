import pandas as pd
import numpy as np
import ccxt
import os
import time
from datetime import datetime

print("="*60)
print(" 🚀 PAPER TRADING TRACKER - Esecuzione Automatica (Hourly Regime)")
print("="*60)

# ==========================================
# CONFIGURAZIONE STRATEGIA
# ==========================================
ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
ASSET_NAMES = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']

# Parametri Regime Detector (Hourly)
Z_THRESH = 2.0
VOL_H = 1.4
VOL_L = 0.7

# Parametri Mean Reversion (Daily)
MR_Z_THRESH = -2.5

# Allocazione per Regime
ALLOCAZIONE = {
    'EXPANSION':   {'tsmom': 0.60, 'mr': 0.00, 'funding': 0.40},
    'NEUTRAL':     {'tsmom': 0.45, 'mr': 0.10, 'funding': 0.45},
    'CONTRACTION': {'tsmom': 0.00, 'mr': 0.10, 'funding': 0.50}
}

# ==========================================
# 1. SCARICA DATI HOURLY (per Regime Detector)
# ==========================================
def fetch_hourly_data(symbol):
    """Scarica dati hourly da Kraken (circa 1000 ore per coprire finestra 720)"""
    exchange = ccxt.kraken({'enableRateLimit': True})
    all_bars = []
    # Kraken limite 720, quindi facciamo un loop per averne ~1000
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
        
    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    return df['close'].tail(1000)

# ==========================================
# 2. SCARICA DATI DAILY (per TSMOM, Drawdown, MR)
# ==========================================
def fetch_daily_data():
    """Scarica dati daily da Kraken"""
    print("📥 Download dati daily da Kraken...")
    exchange = ccxt.kraken({'enableRateLimit': True})
    
    prices = {}
    for symbol in ASSETS:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=720)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.normalize()
            df.set_index('timestamp', inplace=True)
            prices[symbol.replace('/', '')] = df['close']
            print(f"   ✅ Scaricati {len(df)} giorni per {symbol}")
        except Exception as e:
            print(f"   ❌ Errore per {symbol}: {e}")
    
    if not prices:
        raise Exception("Nessun dato scaricato da Kraken")
    
    close_daily = pd.DataFrame(prices)
    close_daily.dropna(inplace=True)
    print(f"   ✅ Totale daily: {len(close_daily)} giorni")
    return close_daily

# ==========================================
# 3. CALCOLA REGIME (HOURLY) E PESI (DAILY)
# ==========================================
def calculate_weights(close_daily, close_hourly_dict):
    """Calcola regime (hourly) e pesi target (daily)"""
    
    # --- REGIME DETECTOR (HOURLY) ---
    print("📊 Calcolo Regime Detector (dati hourly)...")
    regime_series_list = []
    
    for asset in ASSETS:
        symbol_key = asset.replace('/', '')
        if symbol_key not in close_hourly_dict:
            continue
            
        close_h = close_hourly_dict[symbol_key]
        returns_h = close_h.pct_change(fill_method=None)
        
        vol_short = returns_h.rolling(window=24).std()
        vol_long = returns_h.rolling(window=168).std()
        vol_ratio = vol_short / vol_long
        
        momentum = returns_h.rolling(window=24).sum()
        mom_mean = momentum.rolling(window=720).mean()
        mom_std = momentum.rolling(window=720).std().replace(0, np.nan)
        z_score = (momentum - mom_mean) / mom_std
        
        is_expansion = (vol_ratio > VOL_H) & (z_score.abs() > Z_THRESH)
        is_contraction = (vol_ratio < VOL_L) & (z_score.abs() < Z_THRESH)
        
        regime_h = pd.Series('NEUTRAL', index=returns_h.index)
        regime_h.loc[is_expansion] = 'EXPANSION'
        regime_h.loc[is_contraction] = 'CONTRACTION'
        
        # Resampling a daily (moda del giorno)
        daily_reg = regime_h.resample('D').agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'NEUTRAL'
        )
        regime_series_list.append(daily_reg)
    
    # Media dei regimi (o prendiamo l'ultimo giorno comune)
    if regime_series_list:
        # Allineiamo gli indici e prendiamo la moda tra gli asset per ogni giorno
        regime_df = pd.DataFrame(regime_series_list).T
        # Prendiamo l'ultimo giorno completo
        last_day_regime = regime_df.iloc[-1].mode().iloc[0]
        regime = last_day_regime
    else:
        regime = 'NEUTRAL'
        
    print(f"   ✅ Regime giornaliero: {regime}")

    # --- TSMOM, DRAWDOWN, MEAN REVERSION (DAILY) ---
    returns_daily = close_daily.pct_change(fill_method=None)
    
    # TSMOM
    returns_30d = close_daily.pct_change(30, fill_method=None)
    raw_signal = (returns_30d.iloc[-1] > 0).astype(int)
    
    # Drawdown Overlay
    first_valid_price = close_daily.apply(lambda x: x.dropna().iloc[0])
    normalized_close = close_daily / first_valid_price
    market_index = normalized_close.mean(axis=1, skipna=True)
    
    rolling_max = market_index.rolling(window=90, min_periods=1).max()
    drawdown = (market_index.iloc[-1] / rolling_max.iloc[-1]) - 1
    overlay = 0.5 if drawdown < -0.15 else 1.0
    
    # Mean Reversion
    mr_zscore = (close_daily.iloc[-1] - close_daily.rolling(14).mean().iloc[-1]) / close_daily.rolling(14).std().iloc[-1]
    sma_50 = close_daily.rolling(50).mean().iloc[-1]
    mr_signal = ((mr_zscore < MR_Z_THRESH) & (close_daily.iloc[-1] > sma_50)).astype(int)
    
    # --- POSITION SIZING ---
    alloc = ALLOCAZIONE[regime]
    weights = {asset: 0.0 for asset in ASSET_NAMES}
    
    if raw_signal.sum() > 0:
        tsmom_w = (raw_signal / raw_signal.sum()) * alloc['tsmom'] * overlay
        for i, asset in enumerate(ASSET_NAMES):
            weights[asset] += tsmom_w.iloc[i]
            
    if mr_signal.sum() > 0:
        mr_w = (mr_signal / mr_signal.sum()) * alloc['mr']
        for i, asset in enumerate(ASSET_NAMES):
            weights[asset] += mr_w.iloc[i]
            
    total_exposure = sum(weights.values())
    weights['CASH'] = max(0.0, 1.0 - total_exposure - alloc['funding'])
    
    return weights, regime, drawdown

# ==========================================
# 4. SALVA RISULTATI
# ==========================================
def save_results(weights, regime, drawdown):
    """Salva i risultati nel CSV"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    row = {
        'date': today,
        'regime': regime,
        'drawdown': round(drawdown, 4),
        'BTCUSDT': round(weights['BTCUSDT'], 4),
        'ETHUSDT': round(weights['ETHUSDT'], 4),
        'SOLUSDT': round(weights['SOLUSDT'], 4),
        'XRPUSDT': round(weights['XRPUSDT'], 4),
        'CASH': round(weights['CASH'], 4)
    }
    
    csv_file = 'paper_trading_log.csv'
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        if today in df_existing['date'].values:
            print(f"   ⚠️ Dati per {today} già presenti, aggiornamento...")
            df_existing = df_existing[df_existing['date'] != today]
        df_combined = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_combined = pd.DataFrame([row])
    
    df_combined.to_csv(csv_file, index=False)
    print(f"   ✅ Salvato in {csv_file}")
    
    print(f"\n📊 RISULTATI PER {today}:")
    print(f"   Regime: {regime}")
    print(f"   Drawdown: {drawdown:.2%}")
    print(f"   Pesi:")
    for asset in ASSET_NAMES + ['CASH']:
        print(f"     {asset}: {weights[asset]:.1%}")

# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================
if __name__ == "__main__":
    try:
        # 1. Scarica dati daily
        close_daily = fetch_daily_data()
        
        # 2. Scarica dati hourly per ogni asset
        print("📥 Download dati hourly da Kraken (per Regime Detector)...")
        close_hourly_dict = {}
        for symbol in ASSETS:
            try:
                close_hourly_dict[symbol.replace('/', '')] = fetch_hourly_data(symbol)
                print(f"   ✅ Scaricati hourly per {symbol}")
            except Exception as e:
                print(f"   ❌ Errore hourly per {symbol}: {e}")
                
        # 3. Calcola pesi
        weights, regime, drawdown = calculate_weights(close_daily, close_hourly_dict)
        
        # 4. Salva
        save_results(weights, regime, drawdown)
        print("\n✅ Paper trading tracker completato!")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        raise
