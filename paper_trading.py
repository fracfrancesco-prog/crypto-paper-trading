import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime

print("="*60)
print(" 🚀 PAPER TRADING TRACKER - Esecuzione Automatica")
print("="*60)

# ==========================================
# CONFIGURAZIONE STRATEGIA
# ==========================================
ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
ASSET_NAMES = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']

# Parametri Regime Detector
Z_THRESH = 2.0
VOL_H = 1.4
VOL_L = 0.7

# Parametri Mean Reversion
MR_Z_THRESH = -2.5

# Allocazione per Regime
ALLOCAZIONE = {
    'EXPANSION':   {'tsmom': 0.60, 'mr': 0.00, 'funding': 0.40},
    'NEUTRAL':     {'tsmom': 0.45, 'mr': 0.10, 'funding': 0.45},
    'CONTRACTION': {'tsmom': 0.00, 'mr': 0.10, 'funding': 0.50}
}

# ==========================================
# 1. SCARICA DATI DA KRAKEN
# ==========================================
def fetch_data():
    """Scarica dati daily da Kraken"""
    print("📥 Download dati da Kraken...")
    exchange = ccxt.kraken({'enableRateLimit': True})
    
    prices = {}
    for symbol in ASSETS:
        try:
            # Kraken ha un limite di 720 candele
            bars = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=720)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # CORREZIONE: .dt.normalize() è il modo corretto per le Series in pandas
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
    print(f"   ✅ Totale: {len(close_daily)} giorni di dati")
    return close_daily

# ==========================================
# 2. CALCOLA REGIME E PESI
# ==========================================
def calculate_weights(close_daily):
    """Calcola regime e pesi target"""
    returns_daily = close_daily.pct_change(fill_method=None)
    market_returns = returns_daily.mean(axis=1, skipna=True)
    
    # --- REGIME DETECTOR (adattato per dati daily) ---
    vol_short = market_returns.rolling(window=21).std()   # 21 giorni ≈ 1 mese
    vol_long = market_returns.rolling(window=126).std()   # 126 giorni ≈ 6 mesi
    vol_ratio = vol_short / vol_long
    
    momentum = market_returns.rolling(window=21).sum()
    mom_mean = momentum.rolling(window=126).mean()
    mom_std = momentum.rolling(window=126).std().replace(0, np.nan)
    z_score = (momentum - mom_mean) / mom_std
    
    is_expansion = (vol_ratio > VOL_H) & (z_score.abs() > Z_THRESH)
    is_contraction = (vol_ratio < VOL_L) & (z_score.abs() < Z_THRESH)
    
    regime = 'NEUTRAL'
    if is_expansion.iloc[-1]:
        regime = 'EXPANSION'
    elif is_contraction.iloc[-1]:
        regime = 'CONTRACTION'
        
    # --- TSMOM ---
    returns_30d = close_daily.pct_change(30, fill_method=None)
    raw_signal = (returns_30d.iloc[-1] > 0).astype(int)
    
    # --- DRAWDOWN OVERLAY ---
    first_valid_price = close_daily.apply(lambda x: x.dropna().iloc[0])
    normalized_close = close_daily / first_valid_price
    market_index = normalized_close.mean(axis=1, skipna=True)
    
    rolling_max = market_index.rolling(window=90, min_periods=1).max()
    drawdown = (market_index.iloc[-1] / rolling_max.iloc[-1]) - 1
    overlay = 0.5 if drawdown < -0.15 else 1.0
    
    # --- MEAN REVERSION ---
    mr_zscore = (close_daily.iloc[-1] - close_daily.rolling(14).mean().iloc[-1]) / close_daily.rolling(14).std().iloc[-1]
    sma_50 = close_daily.rolling(50).mean().iloc[-1]
    mr_signal = ((mr_zscore < MR_Z_THRESH) & (close_daily.iloc[-1] > sma_50)).astype(int)
    
    # --- POSITION SIZING ---
    alloc = ALLOCAZIONE[regime]
    weights = {asset: 0.0 for asset in ASSET_NAMES}
    
    # TSMOM Weights
    if raw_signal.sum() > 0:
        tsmom_w = (raw_signal / raw_signal.sum()) * alloc['tsmom'] * overlay
        for i, asset in enumerate(ASSET_NAMES):
            weights[asset] += tsmom_w.iloc[i]
            
    # MR Weights
    if mr_signal.sum() > 0:
        mr_w = (mr_signal / mr_signal.sum()) * alloc['mr']
        for i, asset in enumerate(ASSET_NAMES):
            weights[asset] += mr_w.iloc[i]
            
    total_exposure = sum(weights.values())
    weights['CASH'] = max(0.0, 1.0 - total_exposure - alloc['funding'])
    
    return weights, regime, drawdown

# ==========================================
# 3. SALVA RISULTATI
# ==========================================
def save_results(weights, regime, drawdown):
    """Salva i risultati nel CSV"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Prepara la riga da salvare
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
    
    # Se il file esiste, leggi i dati precedenti
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        # Evita duplicati (se lo script gira due volte nello stesso giorno)
        if today in df_existing['date'].values:
            print(f"   ⚠️ Dati per {today} già presenti, aggiornamento...")
            df_existing = df_existing[df_existing['date'] != today]
        df_combined = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_combined = pd.DataFrame([row])
        
    # Salva il CSV
    df_combined.to_csv(csv_file, index=False)
    print(f"   ✅ Salvato in {csv_file}")
    
    # Stampa riepilogo
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
        close_data = fetch_data()
        weights, regime, drawdown = calculate_weights(close_data)
        save_results(weights, regime, drawdown)
        print("\n✅ Paper trading tracker completato!")
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        raise
