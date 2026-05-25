"""
NEPSE Trend Predictor - Flask Web App
"""
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

# ── Config ──────────────────────────────────────────
LOOK_BACK   = 60
EPOCHS      = 40
BATCH_SIZE  = 32
PREDICT_DAYS = 7

STOCKS = {
    "NABIL":  "Nabil Bank",
    "NICA":   "NIC Asia Bank",
    "SCB":    "Standard Chartered Bank Nepal",
    "UPPER":  "Upper Tamakoshi Hydropower",
    "CHCL":   "Chilime Hydropower",
    "NIFRA":  "Nepal Infrastructure Bank",
    "GBIME":  "Global IME Bank",
    "HIDCL":  "HIDCL",
}

FEATURES = [
    "Close","Open","High","Low","Volume",
    "MA7","MA21","MA50",
    "RSI","MACD","MACD_Signal",
    "BB_Upper","BB_Lower",
    "Volatility","Volume_Change","Price_Change"
]

# ── Data ─────────────────────────────────────────────
def fetch_sharesansar(symbol):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://www.sharesansar.com/company/{symbol}",
    }
    records = []
    try:
        for page in range(1, 20):
            url = (f"https://www.sharesansar.com/company/price-history/{symbol}"
                   f"?draw=1&start={(page-1)*100}&length=100&_=1")
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200: break
            data = resp.json()
            rows = data.get("data", [])
            if not rows: break
            for row in rows:
                try:
                    records.append({
                        "Date":   pd.to_datetime(row[1]),
                        "Open":   float(str(row[2]).replace(",","")),
                        "High":   float(str(row[3]).replace(",","")),
                        "Low":    float(str(row[4]).replace(",","")),
                        "Close":  float(str(row[5]).replace(",","")),
                        "Volume": float(str(row[7]).replace(",","") or 0),
                    })
                except: continue
            if len(rows) < 100: break
        if records:
            df = pd.DataFrame(records)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            return df, "live"
    except: pass
    return None, None

def generate_synthetic(symbol, days=600):
    np.random.seed(abs(hash(symbol)) % (2**31))
    dates = pd.date_range(end=datetime.today(), periods=days, freq='B')
    price = 300.0
    prices = []
    volumes = []
    for i in range(days):
        trend = np.random.choice([-1,1], p=[0.45,0.55]) * 0.0005
        vol   = np.random.normal(0, 0.015)
        mr    = -0.01 * (price - 300) / 300
        price = max(price * (1 + trend + vol + mr), 50)
        prices.append(price)
        volumes.append(int(np.random.lognormal(10, 1)))
    closes = np.array(prices)
    df = pd.DataFrame({
        "Open":   closes * (1 + np.random.normal(0,0.005,days)),
        "High":   closes * (1 + np.abs(np.random.normal(0,0.008,days))),
        "Low":    closes * (1 - np.abs(np.random.normal(0,0.008,days))),
        "Close":  closes,
        "Volume": volumes,
    }, index=dates)
    return df, "synthetic"

def get_data(symbol):
    df, source = fetch_sharesansar(symbol)
    if df is None or len(df) < 150:
        df, source = generate_synthetic(symbol)
    return df, source

def add_indicators(df):
    c = df["Close"].copy()
    df = df.copy()
    df["MA7"]  = c.rolling(7).mean()
    df["MA21"] = c.rolling(21).mean()
    df["MA50"] = c.rolling(50).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain/(loss+1e-9)))
    ema12 = c.ewm(span=12,adjust=False).mean()
    ema26 = c.ewm(span=26,adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    r20 = c.rolling(20)
    df["BB_Upper"] = r20.mean() + 2*r20.std()
    df["BB_Lower"] = r20.mean() - 2*r20.std()
    df["Volatility"]    = c.rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Price_Change"]  = c.pct_change()
    df["Target"] = (c.shift(-1) > c).astype(int)
    df.dropna(inplace=True)
    return df

def build_sequences(data, labels, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

def train_predict(symbol):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    df_raw, source = get_data(symbol)
    df = add_indicators(df_raw)

    feat   = df[FEATURES].values
    labels = df["Target"].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feat)

    split = int(len(scaled) * 0.8)
    X_all, y_all = build_sequences(scaled, labels, LOOK_BACK)
    train_size   = split - LOOK_BACK
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_test,  y_test  = X_all[train_size:], y_all[train_size:]

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(LOOK_BACK, len(FEATURES))),
        BatchNormalization(), Dropout(0.3),
        Bidirectional(LSTM(64)),
        BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"), Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    cb = EarlyStopping(patience=8, restore_best_weights=True, verbose=0)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cb], verbose=0)

    y_pred = (model.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Future predictions
    window = scaled[-LOOK_BACK:].copy()
    predictions = []
    last_date  = df.index[-1]
    for i in range(PREDICT_DAYS):
        seq  = window[-LOOK_BACK:].reshape(1, LOOK_BACK, len(FEATURES))
        prob = float(model.predict(seq, verbose=0)[0][0])
        direction  = "UP" if prob >= 0.5 else "DOWN"
        confidence = round(prob*100 if prob>=0.5 else (1-prob)*100, 1)
        next_date  = last_date + timedelta(days=i+1)
        while next_date.weekday() >= 6:
            next_date += timedelta(days=1)
        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "day":  next_date.strftime("%A"),
            "direction": direction,
            "confidence": confidence,
            "prob": round(prob, 4),
        })
        new_row = window[-1].copy()
        ci = FEATURES.index("Close")
        new_row[ci] = np.clip(new_row[ci] + (0.01 if prob>=0.5 else -0.01), 0, 1)
        window = np.vstack([window, new_row])

    # Chart data (last 120 days)
    chart_df = df.tail(120)
    chart_data = {
        "dates":    [str(d.date()) for d in chart_df.index],
        "close":    chart_df["Close"].round(2).tolist(),
        "ma21":     chart_df["MA21"].round(2).tolist(),
        "ma50":     chart_df["MA50"].round(2).tolist(),
        "bb_upper": chart_df["BB_Upper"].round(2).tolist(),
        "bb_lower": chart_df["BB_Lower"].round(2).tolist(),
        "rsi":      chart_df["RSI"].round(2).tolist(),
        "volume":   chart_df["Volume"].tolist(),
        "macd":     chart_df["MACD"].round(2).tolist(),
        "macd_sig": chart_df["MACD_Signal"].round(2).tolist(),
    }

    ups   = sum(1 for p in predictions if p["direction"]=="UP")
    downs = PREDICT_DAYS - ups
    outlook = "BULLISH" if ups > downs else "BEARISH" if downs > ups else "NEUTRAL"
    avg_conf = round(sum(p["confidence"] for p in predictions)/PREDICT_DAYS, 1)

    return {
        "symbol":      symbol,
        "name":        STOCKS.get(symbol, symbol),
        "source":      source,
        "accuracy":    accuracy,
        "latest_close": round(float(df["Close"].iloc[-1]), 2),
        "high_52w":    round(float(df["Close"].tail(252).max()), 2),
        "low_52w":     round(float(df["Close"].tail(252).min()), 2),
        "predictions": predictions,
        "outlook":     outlook,
        "avg_conf":    avg_conf,
        "chart":       chart_data,
        "total_days":  len(df),
        "train_acc":   round(float(history.history["accuracy"][-1])*100, 2),
        "val_acc":     round(float(history.history["val_accuracy"][-1])*100, 2),
        "loss_history": [round(x,4) for x in history.history["loss"]],
        "val_loss_history": [round(x,4) for x in history.history["val_loss"]],
    }

# ── Routes ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", stocks=STOCKS)

@app.route("/predict", methods=["POST"])
def predict():
    symbol = request.json.get("symbol", "NABIL")
    try:
        result = train_predict(symbol)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/stocks")
def stocks():
    return jsonify(STOCKS)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
