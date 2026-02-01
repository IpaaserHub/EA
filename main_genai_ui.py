from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import sqlite3
import datetime
import logging
import statistics
import os

# --- 【重要】AI設定エリア ---

# 1. どちらのAIを使うか選ぶ ("openai" または "google")
ACTIVE_AI_MODEL = "openai" 

# 2. 各社のAPIキー設定 (環境変数から取得、なければプレースホルダー)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key-here")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your-google-api-key-here") 

# 3. 自分の口座ID
ALLOWED_ACCOUNTS = [75449373] 

# --- 設定 ---
DATABASE_NAME = "trading_log.db"
HISTORY_SIZE = 100
ATR_PERIOD = 14

current_persona = "Balanced"
price_history = {} 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Server (UI Ver)", version="3.3.0")

openai_client = None
gemini_model = None

if ACTIVE_AI_MODEL == "openai":
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI Client Initialized")
    except: pass
elif ACTIVE_AI_MODEL == "google":
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Gemini Client Initialized")
    except: pass

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trade_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  account_id INTEGER, symbol TEXT, action TEXT, 
                  price REAL, sl REAL, tp REAL, comment TEXT, persona TEXT)''')
    conn.commit()
    conn.close()

init_db()

class MarketData(BaseModel):
    account_id: int; symbol: str; bid: float; ask: float; bar_time: int; equity: float; daily_profit: float 
class TradeSignal(BaseModel):
    action: str; sl_price: float; tp_price: float; comment: str; server_time: str
class HistoryData(BaseModel):
    account_id: int; symbol: str; prices: List[float]

def calculate_atr(prices, period):
    if len(prices) < period + 1: return 0.01 
    ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    return statistics.mean(ranges[-period:])

def find_high_low(prices):
    if not prices: return 0, 0
    return max(prices), min(prices)

def linear_regression_channel(prices):
    n = len(prices); x = list(range(n)); y = prices
    if n < 2: return 0, prices[-1]
    mean_x = statistics.mean(x); mean_y = statistics.mean(y)
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0
    return slope, mean_y - slope * mean_x

PERSONA_PROMPTS = {
    "Aggressive": "あなたは「超攻撃的なスキャルパー」です。リスクを恐れず、機会損失を最も嫌ってください。迷ったら「GO」を出してください。",
    "Balanced": "あなたは「バランス重視のプロトレーダー」です。リスクとリターンのバランスを見極めてください。",
    "Conservative": "あなたは「極めて慎重な資産運用家」です。100%の自信がある鉄板パターン以外は「STOP」を出してください。"
}

def ask_genai_opinion(symbol, slope, atr, position, trend_type):
    if (ACTIVE_AI_MODEL=="openai" and "sk-" not in OPENAI_API_KEY) or (ACTIVE_AI_MODEL=="google" and "AIza" not in GOOGLE_API_KEY):
        return True, "AI_Skipped"
    try:
        persona_instruction = PERSONA_PROMPTS.get(current_persona, PERSONA_PROMPTS["Balanced"])
        prompt = f"""
        {persona_instruction}
        【データ】通貨:{symbol}, トレンド:{trend_type}({slope:.6f}), 位置:{position*100:.1f}%, ATR:{atr:.3f}
        エントリー判断を GO または STOP の一単語で答えてください。
        """
        answer = ""
        if ACTIVE_AI_MODEL == "openai" and openai_client:
            resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=10)
            answer = resp.choices[0].message.content.strip()
        elif ACTIVE_AI_MODEL == "google" and gemini_model:
            resp = gemini_model.generate_content(prompt)
            answer = resp.text.strip()
        logger.info(f"🤖 AI ({current_persona}): {answer}")
        return ("GO" in answer.upper()), f"{current_persona}_{answer}"
    except Exception as e:
        logger.error(f"AI Error: {e}"); return True, "Error_Pass"

def analyze_market_logic(data: MarketData) -> dict:
    symbol = data.symbol; current_price = data.ask
    if symbol not in price_history: price_history[symbol] = []
    price_history[symbol].append(current_price)
    if len(price_history[symbol]) > HISTORY_SIZE + 10: price_history[symbol].pop(0)
    history = price_history[symbol]
    
    if len(history) < HISTORY_SIZE:
        return {"action": "NO_TRADE", "comment": "Learning...", "sl": 0.0, "tp": 0.0}

    highest, lowest = find_high_low(history)
    price_range = highest - lowest
    position = (current_price - lowest) / price_range if price_range > 0 else 0.5
    slope, _ = linear_regression_channel(history)
    atr = calculate_atr(history, ATR_PERIOD)

    # --- 【修正】性格に合わせてテクニカルの判定基準も変える ---
    buy_thresh = 0.6  # Default
    sell_thresh = 0.4 # Default
    
    if current_persona == "Aggressive":
        buy_thresh = 0.9  # ほぼ天井でもイケイケで買う
        sell_thresh = 0.1 # ほぼ底でもイケイケで売る
    elif current_persona == "Conservative":
        buy_thresh = 0.3  # 深い押し目しか買わない
        sell_thresh = 0.7 # 高い戻り目しか売らない

    logger.info(f"Env: Range={position:.2f} | Slope={slope:.5f} | AI Mode={current_persona} (Thresh:{buy_thresh}/{sell_thresh})")

    signal = "NO_TRADE"; comment = "Wait"; sl=0.0; tp=0.0; trend="None"

    # テクニカル判定
    mid_price = (highest + lowest) / 2

    if slope > 0.0001 and position < buy_thresh:
        signal="BUY"; trend="Up"
        if current_persona == "Aggressive":
            tp = current_price + (atr * 3.0); sl = current_price - (atr * 1.5)
        elif current_persona == "Conservative":
            tp = mid_price; sl = lowest - (atr * 0.2)
        else: # Balanced
            tp = highest - (atr * 0.1); sl = lowest - (atr * 0.5)

    elif slope < -0.0001 and position > sell_thresh:
        signal="SELL"; trend="Down"
        if current_persona == "Aggressive":
            tp = current_price - (atr * 3.0); sl = current_price + (atr * 1.5)
        elif current_persona == "Conservative":
            tp = mid_price; sl = highest + (atr * 0.2)
        else: # Balanced
            tp = lowest + (atr * 0.1); sl = highest + (atr * 0.5)

    if signal != "NO_TRADE":
        approved, ai_msg = ask_genai_opinion(symbol, slope, atr, position, trend)
        if approved: comment = f"{trend}_{ai_msg}"
        else: return {"action": "NO_TRADE", "comment": f"Cancel_{ai_msg}", "sl":0.0, "tp":0.0}

    return {"action": signal, "sl": round(sl,5), "tp": round(tp,5), "comment": comment}

def save_log(data, signal):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO trade_logs (account_id, symbol, action, price, sl, tp, comment, persona) VALUES (?,?,?,?,?,?,?,?)",
                  (data.account_id, data.symbol, signal["action"], data.ask, signal["sl"], signal["tp"], signal["comment"], current_persona))
        conn.commit(); conn.close()
    except: pass

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    conn = sqlite3.connect(DATABASE_NAME); c = conn.cursor()
    try: c.execute("SELECT timestamp, symbol, action, comment, persona FROM trade_logs ORDER BY id DESC LIMIT 10"); logs = c.fetchall()
    except: logs = []
    conn.close()
    rows = "".join([f"<tr><td>{l[0]}</td><td>{l[1]}</td><td>{l[2]}</td><td>{l[3]}</td><td>{l[4] if l[4] else 'Unknown'}</td></tr>" for l in logs])
    return f"""<!DOCTYPE html><html><head><title>AI EA Dashboard</title><style>body{{font-family:sans-serif;padding:20px;background:#f4f4f9;}}.card{{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);margin-bottom:20px;}}.badge{{padding:5px 10px;color:white;border-radius:4px;}}.Aggressive{{background:#e74c3c;}}.Balanced{{background:#3498db;}}.Conservative{{background:#27ae60;}}table{{width:100%;border-collapse:collapse;}}th,td{{border:1px solid #ddd;padding:8px;}}</style></head><body><h1>🤖 AI EA Dashboard</h1><div class="card">Current Mode: <span class="badge {current_persona}">{current_persona}</span><br><br><form action="/set_persona" method="post"><button type="submit" name="mode" value="Aggressive">🔥 Aggressive</button> <button type="submit" name="mode" value="Balanced">⚖️ Balanced</button> <button type="submit" name="mode" value="Conservative">🛡️ Conservative</button></form></div><div class="card"><table><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Comment</th><th>Persona</th></tr>{rows}</table></div></body></html>"""

@app.post("/set_persona", response_class=RedirectResponse)
async def set_persona(request: Request):
    global current_persona
    form_data = await request.form()
    current_persona = form_data.get("mode")
    logger.info(f"🔄 Persona Changed to: {current_persona}")
    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/history")
def update_history(data: HistoryData):
    if data.account_id not in ALLOWED_ACCOUNTS: return {"status": "error"}
    price_history[data.symbol] = data.prices
    logger.info(f"Loaded History: {len(data.prices)} bars")
    return {"status": "ok"}

@app.post("/signal", response_model=TradeSignal)
def get_signal(data: MarketData):
    if data.account_id not in ALLOWED_ACCOUNTS: return {"action": "NO_TRADE", "sl_price": 0, "tp_price": 0, "comment": "License Invalid", "server_time": str(datetime.datetime.now())}
    result = analyze_market_logic(data)
    save_log(data, result)
    return {"action": result["action"], "sl_price": result["sl"], "tp_price": result["tp"], "comment": result["comment"], "server_time": str(datetime.datetime.now())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)