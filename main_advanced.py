from fastapi import FastAPI
from pydantic import BaseModel
from typing import List 
import uvicorn
import sqlite3
import datetime
import logging
import statistics
import math

# --- 戦略設定 ---
DATABASE_NAME = "trading_log.db"
# ここに自分の口座IDを入れてください
ALLOWED_ACCOUNTS = [75449373] 
NEWS_BLOCK_HOURS = []

# ロジック用パラメータ
HISTORY_SIZE = 100     # 過去100本分のデータを見て環境認識する
ATR_PERIOD = 14        # ボラティリティ計算用期間
FIBO_LEVEL = 0.618     # フィボナッチリトレースメント

price_history = {} 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Server (Advanced)", version="2.1.0")

# --- DB初期化 ---
def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trade_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  account_id INTEGER, symbol TEXT, action TEXT, 
                  price REAL, sl REAL, tp REAL, comment TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- データモデル ---
class MarketData(BaseModel):
    account_id: int
    symbol: str
    bid: float
    ask: float
    bar_time: int
    equity: float       
    daily_profit: float 

class TradeSignal(BaseModel):
    action: str      
    sl_price: float
    tp_price: float
    comment: str
    server_time: str

# 過去データ受け取り用のモデル
class HistoryData(BaseModel):
    account_id: int
    symbol: str
    prices: List[float]

# --- 高度な計算ロジック ---

def calculate_atr(prices: list, period: int) -> float:
    if len(prices) < period + 1:
        return 0.01 
    ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    return statistics.mean(ranges[-period:])

def find_high_low(prices: list):
    if not prices:
        return 0, 0
    return max(prices), min(prices)

def linear_regression_channel(prices: list):
    n = len(prices)
    if n < 2:
        return 0, prices[-1]
    x = list(range(n))
    y = prices
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0
    intercept = mean_y - slope * mean_x
    current_theoretical_price = slope * (n - 1) + intercept
    return slope, current_theoretical_price

# --- メインロジック ---

def analyze_market_logic(data: MarketData) -> dict:
    symbol = data.symbol
    current_price = data.ask

    # データ蓄積
    if symbol not in price_history:
        price_history[symbol] = []
    
    price_history[symbol].append(current_price)
    
    if len(price_history[symbol]) > HISTORY_SIZE + 10:
        price_history[symbol].pop(0)

    history = price_history[symbol]
    
    # データ不足時は待機
    if len(history) < HISTORY_SIZE:
        return {"action": "NO_TRADE", "comment": f"Learning... ({len(history)}/{HISTORY_SIZE})"}

    # 環境認識
    highest_price, lowest_price = find_high_low(history)
    price_range = highest_price - lowest_price
    position_in_range = 0.5
    if price_range > 0:
        position_in_range = (current_price - lowest_price) / price_range

    slope, center_line = linear_regression_channel(history)
    atr = calculate_atr(history, ATR_PERIOD)

    logger.info(f"Env: Range={position_in_range:.2f} | Slope={slope:.5f} | ATR={atr:.4f}")

    # エントリー判断
    signal_type = "NO_TRADE"
    comment = "Wait"
    sl = 0.0
    tp = 0.0

    # ロジック1: 上昇チャネルでの押し目買い
    if slope > 0.0001 and position_in_range < 0.4:
        signal_type = "BUY"
        comment = "Trend_Dip_Buy"
        sl = lowest_price - (atr * 0.5)
        tp = highest_price - (atr * 0.5)

    # ロジック2: 下降チャネルでの戻り売り
    elif slope < -0.0001 and position_in_range > 0.6:
        signal_type = "SELL"
        comment = "Trend_Rally_Sell"
        sl = highest_price + (atr * 0.5)
        tp = lowest_price + (atr * 0.5)

    if signal_type != "NO_TRADE":
        risk = abs(current_price - sl)
        reward = abs(tp - current_price)
        if risk == 0 or (reward / risk) < 1.0:
            return {"action": "NO_TRADE", "comment": "Bad_Risk_Reward"}

    return {
        "action": signal_type,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "comment": comment
    }

def check_license(account_id: int) -> bool:
    return account_id in ALLOWED_ACCOUNTS

def save_log(data: MarketData, signal: dict):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO trade_logs (account_id, symbol, action, price, sl, tp, comment) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (data.account_id, data.symbol, signal["action"], data.ask, signal["sl"], signal["tp"], signal["comment"]))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Error: {e}")

# --- API ---

@app.get("/")
def root():
    return {"status": "running"}

# 起動時に過去データを受け取るエンドポイント
@app.post("/history")
def update_history(data: HistoryData):
    if not check_license(data.account_id):
        return {"status": "error", "message": "License Invalid"}
    
    # 履歴を上書き保存
    price_history[data.symbol] = data.prices
    logger.info(f"Loaded History for {data.symbol}: {len(data.prices)} bars")
    return {"status": "ok", "loaded_count": len(data.prices)}

@app.post("/signal", response_model=TradeSignal)
def get_signal(data: MarketData):
    if not check_license(data.account_id):
        return TradeSignal(action="NO_TRADE", sl_price=0, tp_price=0, comment="License Invalid", server_time=str(datetime.datetime.now()))

    result = analyze_market_logic(data)
    save_log(data, result)
    
    return TradeSignal(
        action=result.get("action", "NO_TRADE"),
        sl_price=result.get("sl", 0.0),
        tp_price=result.get("tp", 0.0),
        comment=result.get("comment", ""),
        server_time=str(datetime.datetime.now())
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)