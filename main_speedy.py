from fastapi import FastAPI
from pydantic import BaseModel
from typing import List 
import uvicorn
import sqlite3
import datetime
import logging
import statistics

# --- 設定 ---
DATABASE_NAME = "trading_log.db"
# 【重要】自分の口座IDを入れてください
ALLOWED_ACCOUNTS = [123456, 111222, 999888, 75449373] 
NEWS_BLOCK_HOURS = []

# 【お祭り設定】判定基準を極端に緩くしています
HISTORY_SIZE = 5       # 過去5本だけで判断（すぐ分析完了）
ATR_PERIOD = 5         # 感度ビンビン
SLOPE_THRESHOLD = 0.0  # 傾きが0でなければOK（ほぼ常にトレンド判定）

price_history = {} 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Server (Speedy Test)", version="3.2.0")

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

class HistoryData(BaseModel):
    account_id: int
    symbol: str
    prices: List[float]

# --- ロジック ---
def calculate_atr(prices: list, period: int) -> float:
    if len(prices) < period + 1: return 0.01 
    ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    return statistics.mean(ranges[-period:])

def linear_regression_channel(prices: list):
    n = len(prices)
    if n < 2: return 0
    x = list(range(n))
    y = prices
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0
    return slope

def analyze_market_logic(data: MarketData) -> dict:
    symbol = data.symbol
    current_price = data.ask

    if symbol not in price_history: price_history[symbol] = []
    price_history[symbol].append(current_price)
    if len(price_history[symbol]) > HISTORY_SIZE + 5: price_history[symbol].pop(0)
    history = price_history[symbol]
    
    # データ不足時
    if len(history) < HISTORY_SIZE:
        return {"action": "NO_TRADE", "comment": f"Waiting... ({len(history)}/{HISTORY_SIZE})"}

    # 分析
    slope = linear_regression_channel(history)
    atr = calculate_atr(history, ATR_PERIOD)

    logger.info(f"Speedy Analysis: Slope={slope:.6f} | ATR={atr:.4f}")

    signal_type = "NO_TRADE"
    comment = "Wait"
    sl = 0.0
    tp = 0.0

    # 【激甘ロジック・確実発注版】
    # BTCJPYなどのエラー4756（StopsLevel違反）を回避するため、
    # 損切り(SL)と利確(TP)をあえて「0.0（設定なし）」にして発注します。
    # これで注文自体は確実に通ります。
    if slope > SLOPE_THRESHOLD:
        signal_type = "BUY"
        comment = "Speedy_Buy_NoSL"
        sl = 0.0 
        tp = 0.0 

    elif slope < -SLOPE_THRESHOLD:
        signal_type = "SELL"
        comment = "Speedy_Sell_NoSL"
        sl = 0.0
        tp = 0.0

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
def root(): return {"status": "running"}

@app.post("/history")
def update_history(data: HistoryData):
    if not check_license(data.account_id): return {"status": "error"}
    price_history[data.symbol] = data.prices[-HISTORY_SIZE:] # 必要な分だけ保持
    logger.info(f"Loaded History: {len(data.prices)} bars. Ready to trade!")
    return {"status": "ok"}

@app.post("/signal", response_model=TradeSignal)
def get_signal(data: MarketData):
    if not check_license(data.account_id):
        return TradeSignal(action="NO_TRADE", sl_price=0, tp_price=0, comment="License Invalid", server_time=str(datetime.datetime.now()))

    result = analyze_market_logic(data)
    save_log(data, result)
    return TradeSignal(action=result.get("action", "NO_TRADE"), sl_price=result.get("sl", 0.0), tp_price=result.get("tp", 0.0), comment=result.get("comment", ""), server_time=str(datetime.datetime.now()))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)