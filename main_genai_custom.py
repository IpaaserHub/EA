from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sqlite3
import datetime
import logging
import statistics
import os
import json
import re

# --- AI Modules (built 2026-02-06) ---
from src.ai.news_filter import NewsFilter
from src.ai.regime_detector import RegimeDetector
from src.optimizer.report_generator import ReportGenerator

# --- 【重要】AI設定エリア ---

# 1. どちらのAIを使うか選ぶ ("openai" または "google")
ACTIVE_AI_MODEL = "openai" 

# 2. 各社のAPIキー設定 (環境変数から取得、なければプレースホルダー)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key-here")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your-google-api-key-here") 

# 3. 自分の口座ID (例: 75449373)
ALLOWED_ACCOUNTS = [75449373, 75480718]

# --- 設定 ---
DATABASE_NAME = "trading_log.db"
HISTORY_SIZE = 100  # デフォルト値（銘柄別設定で上書き）
ATR_PERIOD = 14

# --- 銘柄別設定（ガイド準拠） ---
SYMBOL_CONFIG = {
    "BTCJPY":  {"history_size": 120, "max_positions": 1, "cooldown_minutes": 30, "atr_multiplier": 0.7},
    "BTCUSD":  {"history_size": 120, "max_positions": 1, "cooldown_minutes": 30, "atr_multiplier": 0.7},
    "XAUJPY":  {"history_size": 100, "max_positions": 2, "cooldown_minutes": 15, "atr_multiplier": 1.0},
    "XAUUSD":  {"history_size": 100, "max_positions": 2, "cooldown_minutes": 15, "atr_multiplier": 1.0},
    "GBPJPY":  {"history_size": 75,  "max_positions": 2, "cooldown_minutes": 10, "atr_multiplier": 1.0},
    "GBPUSD":  {"history_size": 75,  "max_positions": 2, "cooldown_minutes": 10, "atr_multiplier": 1.0},
    "USDJPY":  {"history_size": 100, "max_positions": 2, "cooldown_minutes": 10, "atr_multiplier": 1.0},
    "EURUSD":  {"history_size": 100, "max_positions": 2, "cooldown_minutes": 10, "atr_multiplier": 1.0},
    "EURJPY":  {"history_size": 100, "max_positions": 2, "cooldown_minutes": 10, "atr_multiplier": 1.0},
}
DEFAULT_SYMBOL_CONFIG = {"history_size": 100, "max_positions": 2, "cooldown_minutes": 10, "atr_multiplier": 1.0}

# --- クールダウン管理 ---
# {symbol: {"last_loss_time": datetime, "consecutive_losses": int}}
cooldown_state = {}
MAX_CONSECUTIVE_LOSSES = 3  # 連続損失でクールダウン発動

# デフォルト設定 (Balanced)
PERSONA_PROMPTS = ["Aggressive", "Balanced", "Conservative"]
current_settings = {
    "persona_name": "Balanced",
    "buy_thresh": 0.6,
    "sell_thresh": 0.4
}

# ============================================================
# v10.0: 高勝率版エントリーパラメータ
# ============================================================
# v10.5: デュアルモード対応エントリーパラメータ
# STABLE: 安定重視（M15/H1）- 低頻度・高勝率
# AGGRESSIVE: 収益重視（M5）- 高頻度・アフィリエイト増
# ============================================================

# 現在のトレードモード（"STABLE" または "AGGRESSIVE"）
TRADE_MODE = "AGGRESSIVE"

# 安定モード: M15/H1用（日2-3回、高勝率）
ENTRY_PARAMS_STABLE = {
    "XAUJPY": {
        "adx_threshold": 15,
        "slope_threshold": 0.00004,
        "buy_position": 0.48,
        "sell_position": 0.52,
        "rsi_buy_max": 70,
        "rsi_sell_min": 30,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.5,
        "sl_mult": 1.5,
    },
    "DEFAULT": {
        "adx_threshold": 20,
        "slope_threshold": 0.00008,
        "buy_position": 0.45,
        "sell_position": 0.55,
        "rsi_buy_max": 65,
        "rsi_sell_min": 35,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.5,
        "sl_mult": 1.5,
    }
}

# 積極モード: M5用（日20回+、アフィリエイト重視）
ENTRY_PARAMS_AGGRESSIVE = {
    "XAUJPY": {
        "adx_threshold": 5,         # v10.6: 10→5に緩和（PF 1.12→1.41）
        "slope_threshold": 0.00001, # v10.6: 0.00002→0.00001に緩和
        "buy_position": 0.50,
        "sell_position": 0.50,
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.0,
        "sl_mult": 1.5,             # v10.7: 1.2→1.5（勝率48→53%、利益+8%）
    },
    "DEFAULT": {
        "adx_threshold": 10,
        "slope_threshold": 0.00002,
        "buy_position": 0.50,
        "sell_position": 0.50,
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
        "rsi_extreme_avoid": False,
        "tp_mult": 2.0,
        "sl_mult": 1.5,             # v10.7: 1.2→1.5
    }
}

CONFIG_PARAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "params")

def _load_optimized_params(symbol: str) -> dict:
    """config/params/{symbol}.jsonから最適化済みパラメータを読み込む"""
    filepath = os.path.join(CONFIG_PARAMS_DIR, f"{symbol}.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            params = json.load(f)
        # 必須キーが全て揃っているか確認
        required = ["adx_threshold", "slope_threshold", "buy_position",
                     "sell_position", "rsi_buy_max", "rsi_sell_min", "tp_mult", "sl_mult"]
        if all(k in params for k in required):
            # rsi_extreme_avoidはoptimizer対象外なのでデフォルト付与
            if "rsi_extreme_avoid" not in params:
                params["rsi_extreme_avoid"] = False
            return params
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load optimized params for {symbol}: {e}")
    return None

def get_entry_params(mode: str = None):
    """トレードモードに応じたパラメータを取得（optimizer出力を優先）"""
    if mode is None:
        mode = TRADE_MODE
    # ベースとなるハードコード値
    if mode == "AGGRESSIVE":
        base = ENTRY_PARAMS_AGGRESSIVE
    else:
        base = ENTRY_PARAMS_STABLE

    # config/params/から最適化済みパラメータを上書き
    result = dict(base)  # shallow copy
    for symbol_file in os.listdir(CONFIG_PARAMS_DIR) if os.path.isdir(CONFIG_PARAMS_DIR) else []:
        if not symbol_file.endswith('.json') or symbol_file == 'optimization_history.json':
            continue
        symbol = symbol_file.replace('.json', '')
        optimized = _load_optimized_params(symbol)
        if optimized:
            result[symbol] = optimized
            logging.getLogger(__name__).info(f"📊 Loaded optimized params for {symbol} (updated: {optimized.get('updated_at', '?')})")

    return result

# 後方互換性のためのエイリアス
ENTRY_PARAMS_V10 = ENTRY_PARAMS_STABLE

price_history = {} 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Server (Ultimate Ver)", version="8.0.0")  # v8.0: 完全ルールベース（ADX+RSI+MA）、AI判断なし

# --- AI Module Instances ---
news_filter = NewsFilter("config/economic_calendar.json")
regime_detectors = {}  # Per-symbol, lazy-fitted: {"USDJPY": RegimeDetector, ...}

# --- ヘルスチェック ---
@app.get("/")
def health_check():
    """サーバー稼働確認用"""
    return {
        "status": "running",
        "version": "v10.7",
        "mode": TRADE_MODE,
        "message": "AI Trading Server is running. Use POST /check_entry or /check_exit for trading."
    }

# --- v7.0: AI分析結果キャッシュ ---
# {symbol: {"analysis": {...}, "timestamp": datetime, "ttl_minutes": 5}}
ai_analysis_cache = {}

# --- AIライブラリ初期化 ---
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

class PositionData(BaseModel):
    ticket: int; symbol: str; type: str; vol: float; open: float; sl: float; tp: float; current: float; profit: float

class MarketData(BaseModel):
    account_id: int; symbol: str; bid: float; ask: float; bar_time: int; equity: float; daily_profit: float
    persona: Optional[str] = None
    positions: List[PositionData] = []

class TradeSignal(BaseModel):
    action: str; sl_price: float; tp_price: float; comment: str; server_time: str
    regime: str = ""          # "TRENDING", "RANGING", "VOLATILE"
    news_status: str = ""     # "" if clear, or "NFP in 45min" etc.
class HistoryData(BaseModel):
    account_id: int; symbol: str; prices: List[float]

# --- Phase 2.3: 決済判断のAI化 ---
class ExitCheckRequest(BaseModel):
    account_id: int
    ticket: int
    symbol: str
    position_type: str  # "BUY" or "SELL"
    open_price: float
    current_price: float
    profit: float
    volume: float
    open_time: int  # Unix timestamp
    sl: float
    tp: float
    prices: List[float] = []  # v7.0: オプションで価格履歴を渡す
    # v9.0追加フィールド
    max_profit_seen: float = 0.0  # 最高到達利益（トレーリング用）
    partial_closed: bool = False  # 分割決済済みフラグ

class ExitCheckResponse(BaseModel):
    action: str  # "HOLD" or "CLOSE" or "PARTIAL_CLOSE" or "MODIFY_SL"
    reason: str
    server_time: str
    # v9.0追加フィールド
    new_sl: float = 0.0  # トレーリングストップ/ブレークイーブン時の新SL
    partial_close: bool = False  # 分割決済フラグ
    partial_ratio: float = 0.0  # 分割決済比率（0.5 = 50%決済）

# --- v7.0: 市場分析リクエスト/レスポンス ---
class AnalyzeRequest(BaseModel):
    account_id: int
    symbol: str
    prices: List[float] = []  # オプション: 価格履歴を直接渡す場合

class AnalyzeResponse(BaseModel):
    symbol: str
    trend: str  # "up", "down", "range"
    strength: int  # 1-10
    volatility: str  # "high", "medium", "low"
    risk_level: str  # "high", "medium", "low"
    recommendation: str
    cached: bool
    server_time: str

# --- Helper Functions ---
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

# --- RSI計算関数 ---
RSI_PERIOD = 14
ADX_PERIOD = 14

def calculate_rsi(prices, period=RSI_PERIOD):
    """RSI（相対力指数）を計算"""
    if len(prices) < period + 1:
        return 50.0  # データ不足時は中立値

    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    # 直近のperiod分を使用
    recent_gains = gains[-period:]
    recent_losses = losses[-period:]

    avg_gain = statistics.mean(recent_gains) if recent_gains else 0
    avg_loss = statistics.mean(recent_losses) if recent_losses else 0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ============================================================
# v8.0: ADX（Average Directional Index）計算
# ============================================================
def calculate_adx(prices, period=ADX_PERIOD):
    """
    ADX（平均方向性指数）を計算
    - ADX < 20: レンジ相場（トレンドなし）→ トレードしない
    - ADX 20-25: 弱いトレンド
    - ADX > 25: 強いトレンド → トレード可
    - ADX > 40: 非常に強いトレンド

    簡略化版: 終値のみから計算
    """
    if len(prices) < period + 1:
        return 20.0  # データ不足時は中立値

    # True Rangeの代わりに価格変動幅を使用
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(prices)):
        # 価格変動
        move = prices[i] - prices[i-1]
        tr = abs(move)
        tr_list.append(tr if tr > 0 else 0.0001)  # ゼロ除算防止

        # +DM / -DM の計算
        if move > 0:
            plus_dm_list.append(move)
            minus_dm_list.append(0)
        else:
            plus_dm_list.append(0)
            minus_dm_list.append(abs(move))

    if len(tr_list) < period:
        return 20.0

    # 平滑化（Wilder's Smoothing）
    def wilders_smooth(data, period):
        smoothed = [sum(data[:period]) / period]
        for i in range(period, len(data)):
            smoothed.append((smoothed[-1] * (period - 1) + data[i]) / period)
        return smoothed

    tr_smooth = wilders_smooth(tr_list, period)
    plus_dm_smooth = wilders_smooth(plus_dm_list, period)
    minus_dm_smooth = wilders_smooth(minus_dm_list, period)

    if not tr_smooth:
        return 20.0

    # +DI / -DI の計算
    dx_list = []
    for i in range(len(tr_smooth)):
        if tr_smooth[i] > 0:
            plus_di = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di = 100 * minus_dm_smooth[i] / tr_smooth[i]

            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = 100 * abs(plus_di - minus_di) / di_sum
                dx_list.append(dx)

    if not dx_list:
        return 20.0

    # ADX = DXの平均
    adx = sum(dx_list[-period:]) / min(period, len(dx_list))
    return adx


def calculate_bollinger_bands(prices, period=20, num_std=2):
    """
    ボリンジャーバンドを計算
    Returns: (upper, middle, lower, bandwidth_pct)
    - bandwidth_pct: バンド幅（ボラティリティ指標）
    """
    if len(prices) < period:
        mid = prices[-1] if prices else 0
        return mid * 1.02, mid, mid * 0.98, 2.0

    recent = prices[-period:]
    middle = statistics.mean(recent)
    std = statistics.stdev(recent)

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    # バンド幅（%）
    bandwidth_pct = ((upper - lower) / middle * 100) if middle > 0 else 2.0

    return upper, middle, lower, bandwidth_pct


def calculate_ma_cross(prices, fast_period=10, slow_period=25):
    """
    移動平均線クロスを計算
    Returns: (fast_ma, slow_ma, is_golden_cross, is_dead_cross, cross_strength)
    """
    if len(prices) < slow_period:
        return 0, 0, False, False, 0

    fast_ma = sum(prices[-fast_period:]) / fast_period
    slow_ma = sum(prices[-slow_period:]) / slow_period

    # 現在の状態
    fast_above = fast_ma > slow_ma

    # 5期前の状態（クロス判定用）
    if len(prices) >= slow_period + 5:
        prev_prices = prices[:-5]
        prev_fast = sum(prev_prices[-fast_period:]) / fast_period
        prev_slow = sum(prev_prices[-slow_period:]) / slow_period
        prev_fast_above = prev_fast > prev_slow

        is_golden_cross = fast_above and not prev_fast_above  # 上抜け
        is_dead_cross = not fast_above and prev_fast_above    # 下抜け
    else:
        is_golden_cross = False
        is_dead_cross = False

    # クロス強度（MAの乖離率）
    cross_strength = abs(fast_ma - slow_ma) / slow_ma * 100 if slow_ma > 0 else 0

    return fast_ma, slow_ma, is_golden_cross, is_dead_cross, cross_strength


# --- 時間帯フィルター ---
# 市場活発時間（サーバー時間=UTC想定、必要に応じて調整）
ACTIVE_SESSIONS = {
    # 東京セッション: 00:00-09:00 UTC (9:00-18:00 JST)
    "tokyo": (0, 9),
    # ロンドンセッション: 07:00-16:00 UTC (16:00-25:00 JST)
    "london": (7, 16),
    # NYセッション: 13:00-22:00 UTC (22:00-07:00 JST)
    "ny": (13, 22),
}

# 銘柄別の推奨セッション
SYMBOL_SESSIONS = {
    "USDJPY": ["tokyo", "london", "ny"],
    "EURJPY": ["tokyo", "london"],
    "GBPJPY": ["london", "ny"],
    "EURUSD": ["london", "ny"],
    "GBPUSD": ["london", "ny"],
    "XAUJPY": ["tokyo", "london", "ny"],  # ゴールドは東京・ロンドン・NY（ほぼ24時間）
    "XAUUSD": ["london", "ny"],
    "BTCJPY": ["tokyo", "london", "ny"],  # BTCは24時間だが主要時間推奨
    "BTCUSD": ["tokyo", "london", "ny"],
}

def is_active_trading_time(symbol: str) -> tuple:
    """取引に適した時間帯かチェック（UTC基準）"""
    now = datetime.datetime.utcnow()
    current_hour = now.hour

    # 銘柄に適したセッションを取得
    sessions = SYMBOL_SESSIONS.get(symbol, ["london", "ny"])  # デフォルトはロンドン・NY

    for session_name in sessions:
        if session_name in ACTIVE_SESSIONS:
            start_hour, end_hour = ACTIVE_SESSIONS[session_name]
            if start_hour <= current_hour < end_hour:
                return True, session_name.capitalize()

    return False, f"OffHours (UTC {current_hour}:00)"

# --- 銘柄別設定取得 ---
def get_symbol_config(symbol: str) -> dict:
    """銘柄に応じた設定を取得"""
    return SYMBOL_CONFIG.get(symbol, DEFAULT_SYMBOL_CONFIG)

# --- クールダウンチェック ---
def is_in_cooldown(symbol: str) -> tuple:
    """クールダウン中かどうかをチェック"""
    if symbol not in cooldown_state:
        return False, ""

    state = cooldown_state[symbol]
    config = get_symbol_config(symbol)

    # 連続損失チェック
    if state.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES:
        last_loss = state.get("last_loss_time")
        if last_loss:
            cooldown_end = last_loss + datetime.timedelta(minutes=config["cooldown_minutes"])
            if datetime.datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.datetime.now()).seconds // 60
                return True, f"Cooldown ({remaining}min left)"
            else:
                # クールダウン終了、リセット
                cooldown_state[symbol] = {"consecutive_losses": 0, "last_loss_time": None}
    return False, ""

def record_trade_result(symbol: str, is_loss: bool):
    """トレード結果を記録（クールダウン管理用）"""
    if symbol not in cooldown_state:
        cooldown_state[symbol] = {"consecutive_losses": 0, "last_loss_time": None}

    if is_loss:
        cooldown_state[symbol]["consecutive_losses"] += 1
        cooldown_state[symbol]["last_loss_time"] = datetime.datetime.now()
        logger.warning(f"⚠️ {symbol}: Loss #{cooldown_state[symbol]['consecutive_losses']}")
    else:
        # 勝ちトレードでリセット
        cooldown_state[symbol]["consecutive_losses"] = 0
        cooldown_state[symbol]["last_loss_time"] = None

# --- ポジション数チェック ---
def count_symbol_positions(symbol: str, positions: list) -> int:
    """指定銘柄の現在ポジション数をカウント"""
    return sum(1 for p in positions if p.symbol == symbol)

def can_open_new_position(symbol: str, positions: list) -> tuple:
    """新規ポジションを開けるかチェック"""
    config = get_symbol_config(symbol)
    current_count = count_symbol_positions(symbol, positions)
    max_positions = config["max_positions"]

    if current_count >= max_positions:
        return False, f"MaxPos ({current_count}/{max_positions})"
    return True, ""

PERSONA_PROMPTS_DICT = {
    "Aggressive": "あなたは「超攻撃的なスキャルパー」です。リスクを恐れず、機会損失を最も嫌ってください。迷ったら「GO」を出してください。",
    "Balanced": "あなたは「バランス重視のプロトレーダー」です。リスクとリターンのバランスを見極めてください。",
    "Conservative": "あなたは「極めて慎重な資産運用家」です。100%の自信がある鉄板パターン以外は「STOP」を出してください。"
}

def ask_genai_opinion(symbol, slope, atr, position, trend_type, persona):
    """旧バージョン: GO/STOP判断のみ（後方互換用に残す）"""
    if (ACTIVE_AI_MODEL=="openai" and "sk-" not in OPENAI_API_KEY) or (ACTIVE_AI_MODEL=="google" and "AIza" not in GOOGLE_API_KEY):
        return True, "AI_Skipped"
    try:
        persona_instruction = PERSONA_PROMPTS_DICT.get(persona, PERSONA_PROMPTS_DICT["Balanced"])
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
        logger.info(f"🤖 AI ({persona}): {answer}")
        return ("GO" in answer.upper()), f"{persona}_{answer}"
    except Exception as e:
        logger.error(f"AI Error: {e}"); return True, "Error_Pass"

def ask_genai_trade_decision(symbol, current_price, highest, lowest, slope, atr, rsi, position, trend_type, persona, fallback_sl, fallback_tp):
    """
    AIにエントリー判断とSL/TPの決定を任せる（v6.2.0新機能）

    Returns:
        tuple: (approved: bool, sl: float, tp: float, comment: str)
    """
    # APIキーチェック
    if (ACTIVE_AI_MODEL=="openai" and "sk-" not in OPENAI_API_KEY) or (ACTIVE_AI_MODEL=="google" and "AIza" not in GOOGLE_API_KEY):
        return True, fallback_sl, fallback_tp, "AI_Skipped"

    try:
        persona_instruction = PERSONA_PROMPTS_DICT.get(persona, PERSONA_PROMPTS_DICT["Balanced"])

        prompt = f"""あなたはプロのFXトレーダーAIです。
{persona_instruction}

【現在の相場データ】
- 通貨ペア: {symbol}
- 現在価格: {current_price:.5f}
- 直近高値: {highest:.5f}
- 直近安値: {lowest:.5f}
- トレンド方向: {trend_type}
- トレンド傾き(Slope): {slope:.6f}
- RSI(14): {rsi:.1f}
- ATR(ボラティリティ): {atr:.5f}
- レンジ内位置: {position*100:.1f}%（0%=安値、100%=高値）

【タスク】
このデータを分析し、以下のJSON形式で回答してください：
```json
{{
  "decision": "GO" または "STOP",
  "sl": 損切り価格（数値）,
  "tp": 利確価格（数値）,
  "reason": "判断理由（20文字以内）"
}}
```

【注意】
- {trend_type}方向のエントリーを想定しています
- SL/TPは現実的な価格を設定してください
- 必ず上記JSON形式のみで回答してください
"""

        answer = ""
        if ACTIVE_AI_MODEL == "openai" and openai_client:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3  # 安定した回答のため低めに設定
            )
            answer = resp.choices[0].message.content.strip()
        elif ACTIVE_AI_MODEL == "google" and gemini_model:
            resp = gemini_model.generate_content(prompt)
            answer = resp.text.strip()

        logger.info(f"🤖 AI Trade Decision ({persona}): {answer[:100]}...")

        # JSONパース
        json_match = re.search(r'\{[^{}]*\}', answer, re.DOTALL)
        if json_match:
            ai_result = json.loads(json_match.group())
            decision = ai_result.get("decision", "STOP").upper()
            ai_sl = float(ai_result.get("sl", fallback_sl))
            ai_tp = float(ai_result.get("tp", fallback_tp))
            reason = ai_result.get("reason", "AI判断")[:30]

            # SL/TPの妥当性チェック
            if trend_type.startswith("Up"):
                # BUYの場合: SL < 現在価格 < TP
                if ai_sl >= current_price or ai_tp <= current_price:
                    logger.warning(f"⚠️ AI SL/TP invalid for BUY, using fallback")
                    ai_sl, ai_tp = fallback_sl, fallback_tp
            else:
                # SELLの場合: TP < 現在価格 < SL
                if ai_sl <= current_price or ai_tp >= current_price:
                    logger.warning(f"⚠️ AI SL/TP invalid for SELL, using fallback")
                    ai_sl, ai_tp = fallback_sl, fallback_tp

            approved = "GO" in decision
            return approved, ai_sl, ai_tp, f"{persona}_{reason}"
        else:
            logger.warning(f"⚠️ AI response not JSON, using fallback")
            return True, fallback_sl, fallback_tp, f"{persona}_ParseError"

    except Exception as e:
        logger.error(f"AI Trade Decision Error: {e}")
        return True, fallback_sl, fallback_tp, "AI_Error"

# ============================================================
# v7.0: AI分析関数（市場コンテキストのみ、判断はしない）
# ============================================================
def ask_ai_market_analysis(symbol: str, prices: list, rsi: float, slope: float, atr: float) -> dict:
    """
    AIに市場分析を依頼（判断はせず、コンテキストのみ提供）

    Returns:
        dict: {"trend": str, "strength": int, "volatility": str, "risk_level": str, "recommendation": str}
    """
    # キャッシュチェック（5分間有効）
    if symbol in ai_analysis_cache:
        cache = ai_analysis_cache[symbol]
        cache_age = (datetime.datetime.now() - cache["timestamp"]).total_seconds() / 60
        if cache_age < cache.get("ttl_minutes", 5):
            logger.info(f"📦 Using cached analysis for {symbol} (age: {cache_age:.1f}min)")
            return {**cache["analysis"], "cached": True}

    # デフォルト値（AI失敗時のフォールバック）
    default_analysis = {
        "trend": "range",
        "strength": 5,
        "volatility": "medium",
        "risk_level": "medium",
        "recommendation": "通常取引可",
        "cached": False
    }

    # APIキーチェック
    if (ACTIVE_AI_MODEL=="openai" and "sk-" not in OPENAI_API_KEY) or (ACTIVE_AI_MODEL=="google" and "AIza" not in GOOGLE_API_KEY):
        return default_analysis

    try:
        # トレンド方向の事前計算
        trend_direction = "上昇" if slope > 0.0001 else "下降" if slope < -0.0001 else "レンジ"

        # ボラティリティの事前計算
        if len(prices) > 20:
            recent_range = max(prices[-20:]) - min(prices[-20:])
            avg_price = sum(prices[-20:]) / 20
            vol_pct = (recent_range / avg_price) * 100
        else:
            vol_pct = 1.0

        prompt = f"""あなたは市場分析AIです。以下のデータを分析し、JSON形式で回答してください。

【{symbol}の現在データ】
- RSI(14): {rsi:.1f}
- トレンド方向: {trend_direction}（傾き: {slope:.6f}）
- ATR: {atr:.5f}
- 直近ボラティリティ: {vol_pct:.2f}%

【回答形式】必ず以下のJSON形式のみで回答:
{{
  "trend": "up" or "down" or "range",
  "strength": 1-10の整数（トレンドの強さ）,
  "volatility": "high" or "medium" or "low",
  "risk_level": "high" or "medium" or "low",
  "recommendation": "簡潔な推奨（20字以内）"
}}

【判断基準】
- trend: slope > 0.0001 → up, slope < -0.0001 → down, else → range
- strength: |slope|の大きさとRSIの極端さで判断
- volatility: ATRと価格変動率で判断
- risk_level: RSI極端値、高ボラ、弱トレンドは高リスク
"""

        answer = ""
        if ACTIVE_AI_MODEL == "openai" and openai_client:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            answer = resp.choices[0].message.content.strip()
        elif ACTIVE_AI_MODEL == "google" and gemini_model:
            resp = gemini_model.generate_content(prompt)
            answer = resp.text.strip()

        logger.info(f"🔍 AI Analysis ({symbol}): {answer[:100]}...")

        # JSONパース
        json_match = re.search(r'\{[^{}]*\}', answer, re.DOTALL)
        if json_match:
            ai_result = json.loads(json_match.group())
            analysis = {
                "trend": ai_result.get("trend", "range"),
                "strength": int(ai_result.get("strength", 5)),
                "volatility": ai_result.get("volatility", "medium"),
                "risk_level": ai_result.get("risk_level", "medium"),
                "recommendation": ai_result.get("recommendation", "分析完了")[:30],
                "cached": False
            }

            # キャッシュに保存
            ai_analysis_cache[symbol] = {
                "analysis": analysis,
                "timestamp": datetime.datetime.now(),
                "ttl_minutes": 5
            }

            return analysis

        return default_analysis

    except Exception as e:
        logger.error(f"AI Analysis Error: {e}")
        return default_analysis


# ============================================================
# v8.0: 完全ルールベース決済判断（AI不使用、ADX使用）
# ============================================================
def rule_based_exit_decision_v8(symbol: str, position_type: str, profit: float,
                                 holding_minutes: int, rsi: float, slope: float,
                                 adx: float, prices: list = None) -> tuple:
    """
    v8.0: 完全ルールベースの決済判断（AI不使用）

    Args:
        adx: ADX値（トレンド強度）
        prices: 価格履歴（ボリンジャーバンド計算用）

    Returns:
        tuple: (should_close: bool, reason: str)
    """
    # ====== ADXベースのトレンド判定 ======
    is_ranging = adx < 20          # レンジ相場
    is_weak_trend = 20 <= adx < 25  # 弱いトレンド
    is_strong_trend = adx >= 25     # 強いトレンド
    is_very_strong = adx >= 40      # 非常に強いトレンド

    # slopeからトレンド方向を判定
    trend_up = slope > 0.00005
    trend_down = slope < -0.00005

    # ポジションとトレンドの整合性
    position_aligned = (
        (trend_up and position_type == "BUY") or
        (trend_down and position_type == "SELL")
    )
    position_against = (
        (trend_up and position_type == "SELL") or
        (trend_down and position_type == "BUY")
    )

    # ====== v8.0.4: ADXベースの閾値設定（シンプル版） ======
    if is_ranging:
        loss_threshold = -100
        profit_threshold = 200
        max_hold_minutes = 20
    elif is_weak_trend:
        loss_threshold = -200
        profit_threshold = 500
        max_hold_minutes = 40
    elif is_very_strong:
        loss_threshold = -400
        profit_threshold = 2000
        max_hold_minutes = 120
    else:  # strong trend
        loss_threshold = -300
        profit_threshold = 1000
        max_hold_minutes = 60

    # シンボル別調整（目標金額別ガイド準拠）
    # Lv.3: XAUJPY/XAUUSD - 月20万〜50万、Aggressive向け
    symbol_adjustments = {
        "BTCJPY": {"loss_mult": 1.5, "profit_mult": 1.5},
        "XAUJPY": {"loss_mult": 1.5, "profit_mult": 2.0},  # ゴールド：損切りやや広め、利確大きく
        "XAUUSD": {"loss_mult": 1.5, "profit_mult": 2.0},  # ゴールド：損切りやや広め、利確大きく
        "USDJPY": {"loss_mult": 0.5, "profit_mult": 0.5},  # USDJPYは控えめ
        "GBPJPY": {"loss_mult": 1.2, "profit_mult": 1.5},  # Lv.2: ポンド円
        "EURUSD": {"loss_mult": 1.0, "profit_mult": 1.0},  # Lv.1: 堅実
    }
    adj = symbol_adjustments.get(symbol, {"loss_mult": 1.0, "profit_mult": 1.0})
    loss_threshold *= adj["loss_mult"]
    profit_threshold *= adj["profit_mult"]

    # ====== 決済ルール（優先順） ======

    # 0. レンジ相場（ADX < 20）での特別ルール
    if is_ranging:
        # レンジ相場で損失 → 早期損切り
        if profit < -50:
            return True, f"レンジ損切ADX{adx:.0f}"
        # レンジ相場で小さな利益 → 早期利確
        if profit >= 100:
            return True, f"レンジ利確{profit:.0f}円"

    # 1. XAUJPY/XAUUSD専用ルール（Lv.3 Aggressive向け - 月20万〜50万目標）
    if symbol in ["XAUJPY", "XAUUSD"]:
        # ゴールドはトレンドが長く続くため、利益を最大限伸ばす
        # 強トレンド順方向で利益中 → 大きく伸ばす
        if is_strong_trend and position_aligned and profit > 50:
            return False, f"GOLD順トレンドHOLD"
        # 非常に強いトレンドで利益中 → さらに伸ばす
        if is_very_strong and profit > 100:
            return False, f"GOLD強トレンドHOLD"
        # 逆トレンド + 損失 → 早期損切り（ATRで振り落とされる前に）
        if profit < -100 and position_against:
            return True, f"GOLD逆トレンド損切"
        # 利益が十分出たら確定（弱トレンド時）
        if profit >= 300 and not is_strong_trend:
            return True, f"GOLD利確{profit:.0f}円"

    # 2. BTCJPY専用ルール（v8.1: FalseHold対策で早期損切り）
    if symbol == "BTCJPY":
        # 逆トレンド + 損失 → 早期損切り（SL到達前に切る）
        if profit < -200 and position_against:
            return True, f"BTC逆トレンド損切"
        # RSI極端値 + 損失 → 早期損切り
        if profit < -150 and ((position_type == "BUY" and rsi > 70) or (position_type == "SELL" and rsi < 30)):
            return True, f"BTC_RSI損切"
        # 10分以上保有 + 損失拡大中 → 損切り
        if holding_minutes >= 10 and profit < -100:
            return True, f"BTC時間損切"
        # 強トレンド順方向で利益 → 利益を伸ばす
        if is_strong_trend and position_aligned and profit > 300:
            return False, f"BTC順トレンドHOLD"
        # 小さな利益でも確定（FalseHold防止）
        if profit >= 200 and not is_very_strong:
            return True, f"BTC早期利確{profit:.0f}円"

    # 2. USDJPY専用ルール（v8.0.2: FalseClose対策で閾値緩和）
    if symbol == "USDJPY":
        # 強トレンド順方向で利益中 → 利益を伸ばす
        if is_strong_trend and position_aligned and profit > 100:
            return False, f"USD順トレンドHOLD"
        # レンジで大きな損失 → 損切り
        if profit < -150 and adx < 25:
            return True, f"USD_ADX損切"
        # 強い逆トレンドで損失 → 損切り
        if profit < -100 and position_against and is_strong_trend:
            return True, f"USD逆方向損切"
        # RSI極端値で大きな損失 → 損切り
        if profit < -150 and (rsi < 20 or rsi > 80):
            return True, f"USD_RSI損切"

    # 2. 強制損切り（大損失）
    if profit < loss_threshold * 1.5:
        return True, f"強制損切{profit:.0f}円"

    # 3. 強制利確（大利益）
    if profit > profit_threshold * 1.5:
        return True, f"強制利確{profit:.0f}円"

    # 4. 逆トレンド + 損失 → 損切り
    if profit < 0 and position_against and is_strong_trend:
        return True, f"逆トレンド損切"

    # 5. 通常損切りライン到達
    if profit < loss_threshold:
        return True, f"損切{profit:.0f}円"

    # 6. 利確条件（弱トレンド or RSI極端）
    if profit >= profit_threshold:
        if not is_strong_trend or rsi < 25 or rsi > 75:
            return True, f"利確{profit:.0f}円"

    # 7. 長時間保有 + 損失 → 損切り
    if holding_minutes > max_hold_minutes and profit < 0:
        return True, f"時間損切{holding_minutes}分"

    # 8. 強トレンド + 順方向 + 利益中 → HOLD（利益を伸ばす）
    if position_aligned and profit > 0 and is_strong_trend:
        return False, f"順トレンドHOLD_ADX{adx:.0f}"

    # デフォルト: HOLD
    return False, f"HOLD_ADX{adx:.0f}"


# ============================================================
# v9.0: 決済判断（トレーリングストップ、ブレークイーブン、分割決済）
# ============================================================
# v9.0 パラメータ設定（シンボル別）
EXIT_PARAMS_V9 = {
    "XAUJPY": {
        # トレーリングストップ（利益を守る）- v10.0調整
        "trailing_start": 15,       # 開始利益（円）- 早期発動
        "trailing_distance": 10,    # 価格からの距離（円）- タイト
        # ブレークイーブン（損失回避）- v10.0調整
        "breakeven_trigger": 10,    # BE発動利益（円）- 早期発動
        "breakeven_buffer": 3,      # 建値+バッファ（円）
        # 分割決済（利益確保）- v10.0調整
        "partial_tp1": 20,          # 第1利確（円）- 早期発動
        "partial_tp2": 50,          # 第2利確（円）
        "partial_ratio": 0.5,       # 第1利確時の決済比率
        # 早期損切り（逆トレンド時のみ効果的）
        "early_loss_threshold": -25,  # 早期損切りライン - タイト
    },
    "XAUUSD": {
        "trailing_start": 80,
        "trailing_distance": 50,
        "breakeven_trigger": 60,
        "breakeven_buffer": 15,
        "partial_tp1": 100,
        "partial_tp2": 250,
        "partial_ratio": 0.5,
        "early_loss_threshold": -80,
    },
    "USDJPY": {
        "trailing_start": 50,
        "trailing_distance": 30,
        "breakeven_trigger": 40,
        "breakeven_buffer": 10,
        "partial_tp1": 60,
        "partial_tp2": 150,
        "partial_ratio": 0.5,
        "early_loss_threshold": -50,
    },
    "BTCJPY": {
        "trailing_start": 150,
        "trailing_distance": 100,
        "breakeven_trigger": 100,
        "breakeven_buffer": 30,
        "partial_tp1": 200,
        "partial_tp2": 500,
        "partial_ratio": 0.5,
        "early_loss_threshold": -100,
    },
    "DEFAULT": {
        "trailing_start": 80,
        "trailing_distance": 50,
        "breakeven_trigger": 60,
        "breakeven_buffer": 15,
        "partial_tp1": 100,
        "partial_tp2": 250,
        "partial_ratio": 0.5,
        "early_loss_threshold": -80,
    }
}

def rule_based_exit_decision_v9(
    symbol: str,
    position_type: str,
    profit: float,
    holding_minutes: int,
    rsi: float,
    slope: float,
    adx: float,
    open_price: float,
    current_price: float,
    current_sl: float,
    max_profit_seen: float = 0,  # 最高到達利益（トレーリング用）
    partial_closed: bool = False,  # 分割決済済みフラグ
    prices: list = None
) -> dict:
    """
    v9.0: 高度な決済判断（トレーリングストップ、ブレークイーブン、分割決済）

    Returns:
        dict: {
            "action": str,  # "HOLD", "CLOSE", "PARTIAL_CLOSE", "MODIFY_SL"
            "reason": str,
            "new_sl": float,  # MODIFY_SL時の新SL値
            "partial_ratio": float,  # PARTIAL_CLOSE時の決済比率
        }
    """
    # パラメータ取得
    params = EXIT_PARAMS_V9.get(symbol, EXIT_PARAMS_V9["DEFAULT"])

    # ADXベースのトレンド判定
    is_ranging = adx < 20
    is_strong_trend = adx >= 25
    is_very_strong = adx >= 40

    # トレンド方向
    trend_up = slope > 0.00005
    trend_down = slope < -0.00005

    # ポジションとトレンドの整合性
    position_aligned = (
        (trend_up and position_type == "BUY") or
        (trend_down and position_type == "SELL")
    )
    position_against = (
        (trend_up and position_type == "SELL") or
        (trend_down and position_type == "BUY")
    )

    result = {
        "action": "HOLD",
        "reason": "",
        "new_sl": 0.0,
        "partial_ratio": 0.0
    }

    # ====== 1. 早期損切り（FalseHold対策最優先） ======
    if profit < params["early_loss_threshold"]:
        # 逆トレンド時はさらに早く損切り
        if position_against:
            result["action"] = "CLOSE"
            result["reason"] = f"v9早期損切{profit:.0f}円(逆トレンド)"
            return result
        # 通常の早期損切り
        result["action"] = "CLOSE"
        result["reason"] = f"v9早期損切{profit:.0f}円"
        return result

    # 強い逆トレンド+損失 → 即損切り
    if profit < -30 and position_against and is_strong_trend:
        result["action"] = "CLOSE"
        result["reason"] = f"v9逆トレンド損切{profit:.0f}円"
        return result

    # ====== 2. ブレークイーブン（利益→建値SLへ） ======
    if profit >= params["breakeven_trigger"] and not partial_closed:
        # BUYの場合: SLを建値より上に移動（損失防止）
        if position_type == "BUY":
            new_sl = open_price + params["breakeven_buffer"]
            if current_sl < new_sl:
                result["action"] = "MODIFY_SL"
                result["reason"] = f"v9_BE(SL→{new_sl:.0f})"
                result["new_sl"] = new_sl
                return result
        # SELLの場合: SLを建値より少し上に移動（損失防止）
        # SELL: SLは上にあるので、建値+バッファに下げる
        else:
            new_sl = open_price + params["breakeven_buffer"]
            if current_sl > new_sl:
                result["action"] = "MODIFY_SL"
                result["reason"] = f"v9_BE(SL→{new_sl:.0f})"
                result["new_sl"] = new_sl
                return result

    # ====== 3. 分割決済（第1利確） ======
    if profit >= params["partial_tp1"] and not partial_closed:
        result["action"] = "PARTIAL_CLOSE"
        result["reason"] = f"v9_分割利確{profit:.0f}円"
        result["partial_ratio"] = params["partial_ratio"]
        return result

    # ====== 4. トレーリングストップ（ATR対応） ======
    if profit >= params["trailing_start"]:
        # ATRベースのトレーリング距離（利用可能な場合）、固定距離をフォールバック
        trail_distance = params["trailing_distance"]
        if prices and len(prices) >= ATR_PERIOD + 1:
            atr_val = calculate_atr(prices, ATR_PERIOD)
            if atr_val > 0:
                trail_distance = atr_val * 1.0  # ATR x 1.0

        # 最高利益更新時、SLを追従
        if profit > max_profit_seen:
            if position_type == "BUY":
                new_sl = current_price - trail_distance
                if new_sl > current_sl:
                    result["action"] = "MODIFY_SL"
                    result["reason"] = f"v9_Trail(SL→{new_sl:.0f})"
                    result["new_sl"] = new_sl
                    return result
            else:  # SELL
                new_sl = current_price + trail_distance
                if new_sl < current_sl:
                    result["action"] = "MODIFY_SL"
                    result["reason"] = f"v9_Trail(SL→{new_sl:.0f})"
                    result["new_sl"] = new_sl
                    return result

    # ====== 5. 第2利確（最終利確） ======
    if profit >= params["partial_tp2"]:
        result["action"] = "CLOSE"
        result["reason"] = f"v9_最終利確{profit:.0f}円"
        return result

    # ====== 6. 既存v8ルール（補助） ======
    # レンジ相場での早期決済
    if is_ranging:
        if profit < -50:
            result["action"] = "CLOSE"
            result["reason"] = f"v9レンジ損切ADX{adx:.0f}"
            return result
        if profit >= 80:
            result["action"] = "CLOSE"
            result["reason"] = f"v9レンジ利確{profit:.0f}円"
            return result

    # 長時間保有+損失
    if holding_minutes > 60 and profit < 0:
        result["action"] = "CLOSE"
        result["reason"] = f"v9時間損切{holding_minutes}分"
        return result

    # RSI極端値での決済
    if profit > 50 and (rsi < 20 or rsi > 80):
        result["action"] = "CLOSE"
        result["reason"] = f"v9_RSI決済{profit:.0f}円"
        return result

    # ====== デフォルト: HOLD ======
    result["reason"] = f"v9_HOLD_ADX{adx:.0f}"
    return result


# v7.0互換用（後方互換）
def rule_based_exit_decision(symbol: str, position_type: str, profit: float,
                              holding_minutes: int, rsi: float, slope: float,
                              ai_context: dict = None) -> tuple:
    """v7.0互換: ai_contextがある場合は古いロジックを使用"""
    # v8.0のロジックに転送（ADXは計算できないので仮の値を使用）
    if ai_context:
        trend_strength = ai_context.get("strength", 5)
        # strengthをADXに変換（1-10 → 10-50）
        fake_adx = 10 + (trend_strength * 4)
    else:
        fake_adx = 25  # デフォルト

    return rule_based_exit_decision_v8(symbol, position_type, profit,
                                        holding_minutes, rsi, slope, fake_adx)


def ask_genai_exit_decision(symbol, position_type, open_price, current_price, profit, holding_minutes, sl, tp, rsi, slope, atr=None):
    """
    AIに決済判断を任せる（Phase 2.3新機能 + v6.5改善）

    v6.5改善:
    - ATRベース動的閾値
    - 時間ベース強制決済
    - トレンド強度フィルター

    Returns:
        tuple: (should_close: bool, reason: str)
    """
    # APIキーチェック
    if (ACTIVE_AI_MODEL=="openai" and "sk-" not in OPENAI_API_KEY) or (ACTIVE_AI_MODEL=="google" and "AIza" not in GOOGLE_API_KEY):
        return False, "AI_Skipped"

    # ====== 【改善1】固定閾値（v6.5最終版） ======
    # シンボル別の固定閾値（v2ベース - 最良結果）
    thresholds = {
        "USDJPY": {"loss": -250, "profit": 1500},
        "BTCJPY": {"loss": -400, "profit": 1000},
        "XAUUSD": {"loss": -350, "profit": 1200},
        "XAUJPY": {"loss": -350, "profit": 1200}  # 円建てゴールド
    }
    default_thresh = {"loss": -300, "profit": 1000}
    thresh = thresholds.get(symbol, default_thresh)
    loss_threshold = thresh["loss"]
    profit_threshold = thresh["profit"]
    logger.info(f"📊 Thresholds: {symbol} Loss={loss_threshold}, Profit={profit_threshold}")

    # ====== 【改善2】強制決済（最小限のルールのみ） ======
    # 大損失のみ強制損切り（閾値×2超）
    if profit < loss_threshold * 2:
        logger.info(f"🚨 Force CLOSE: {symbol} loss {profit:.0f} < threshold {loss_threshold*2:.0f}")
        return True, f"強制損切{profit:.0f}円"

    # 大利益のみ強制利確（閾値×2超）
    if profit > profit_threshold * 2:
        logger.info(f"💰 Force CLOSE: {symbol} profit {profit:.0f} > threshold {profit_threshold*2:.0f}")
        return True, f"強制利確{profit:.0f}円"

    # ====== 【改善3】トレンド強度（AIへの情報提供用） ======
    slope_abs = abs(slope) if slope else 0
    trend_strong = slope_abs > 0.00005
    trend_weak = slope_abs < 0.00002

    try:
        # トレンド方向とポジションの整合性を判定
        trend_direction = "UP" if slope > 0 else "DOWN" if slope < 0 else "RANGE"
        position_aligned = (trend_direction == "UP" and position_type == "BUY") or (trend_direction == "DOWN" and position_type == "SELL")
        position_against = not position_aligned and trend_direction != "RANGE"

        # ※ 弱トレンドでも即CLOSEせず、AIに判断を委ねる

        # 判断に必要な追加情報
        is_losing = profit < 0
        is_range = trend_weak

        # 動的閾値をプロンプトに反映（v6.5: HOLD傾向強化）
        prompt = f"""あなたはFX決済判断AIです。【利益は伸ばし、損失は早く切る】が基本です。

【ポジション状況】
- 通貨ペア: {symbol} | ポジション: {position_type}
- 含み損益: {profit:.0f}円
- 保有時間: {holding_minutes}分
- RSI: {rsi:.0f}
- トレンド: {"順方向" if position_aligned else "逆方向" if position_against else "レンジ"}
- トレンド強度: {"強" if trend_strong else "弱" if trend_weak else "中"}

【動的閾値】
- 損切りライン: {loss_threshold:.0f}円
- 利確ライン: {profit_threshold:.0f}円

【判断ルール】

■ CLOSE条件（いずれかに該当すればCLOSE）
1. 損失{abs(loss_threshold):.0f}円超 → 必ずCLOSE
2. 損失中 AND トレンド逆方向 → CLOSE（逆方向の損失は危険）
3. 利益{profit_threshold:.0f}円以上 AND (トレンド弱 OR RSI極端) → CLOSE

■ HOLD条件（以下に該当すればHOLD）
1. 利益中 AND トレンド順方向 AND 利確ライン未達 → HOLD（もっと伸ばす）
2. 小損失 AND トレンド順方向 → HOLD（回復の可能性あり）
3. レンジ相場 AND 損益ほぼゼロ → HOLD（様子見）

■ デフォルト
- 上記に該当しない場合 → HOLD（迷ったらHOLD）

【現在の状態】
- 損失中?: {"YES" if is_losing else "NO"}
- トレンド順方向?: {"YES" if position_aligned else "NO"}
- 利確条件達成?: {"YES" if profit >= profit_threshold else "NO"}
- 損切条件達成?: {"YES" if profit < loss_threshold else "NO"}

【回答】JSON形式のみ：
{{"decision": "HOLD" or "CLOSE", "reason": "理由10字以内"}}
"""

        answer = ""
        if ACTIVE_AI_MODEL == "openai" and openai_client:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2  # 決済判断は安定性重視
            )
            answer = resp.choices[0].message.content.strip()
        elif ACTIVE_AI_MODEL == "google" and gemini_model:
            resp = gemini_model.generate_content(prompt)
            answer = resp.text.strip()

        logger.info(f"🤖 AI Exit Decision ({symbol}): {answer[:80]}...")

        # JSONパース
        json_match = re.search(r'\{[^{}]*\}', answer, re.DOTALL)
        if json_match:
            ai_result = json.loads(json_match.group())
            decision = ai_result.get("decision", "HOLD").upper()
            reason = ai_result.get("reason", "AI判断")[:20]

            should_close = "CLOSE" in decision
            return should_close, reason
        else:
            logger.warning(f"⚠️ AI exit response not JSON")
            return False, "ParseError"

    except Exception as e:
        logger.error(f"AI Exit Decision Error: {e}")
        return False, "AI_Error"

def save_log(data, result, persona):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO trade_logs (account_id, symbol, action, price, sl, tp, comment, persona) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (data.account_id, data.symbol, result["action"], data.ask, result["sl"], result["tp"], result["comment"], persona))
    conn.commit()
    conn.close()

# Global storage for live positions
current_positions = []

def analyze_market_logic(data: MarketData) -> dict:
    global current_positions
    current_positions = data.positions # Update live positions

    symbol = data.symbol
    config = get_symbol_config(symbol)  # 銘柄別設定を取得
    symbol_history_size = config["history_size"]
    atr_multiplier = config["atr_multiplier"]

    # デバッグログ: 受信したポジション数を表示
    if len(data.positions) > 0:
        logger.info(f"📍 {symbol}: {len(data.positions)} positions | Config: HistSize={symbol_history_size}, MaxPos={config['max_positions']}")

    # 1. 性格（決済ルール）の決定
    use_persona = data.persona if (data.persona and data.persona in PERSONA_PROMPTS) else current_settings["persona_name"]

    # --- クールダウンチェック（3連敗後は一定時間休止） ---
    in_cooldown, cooldown_msg = is_in_cooldown(symbol)
    if in_cooldown:
        logger.info(f"🛑 {symbol}: {cooldown_msg}")
        return {"action": "NO_TRADE", "comment": cooldown_msg, "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": "", "news_status": ""}

    # --- ニュースフィルター（経済イベント前後は取引停止） ---
    can_trade, news_reason = news_filter.should_trade(symbol, datetime.datetime.utcnow())
    news_status_str = news_reason if not can_trade else ""
    if not can_trade:
        logger.info(f"📰 {symbol}: {news_reason}")
        return {"action": "NO_TRADE", "comment": f"News:{news_reason}", "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": "", "news_status": news_status_str}

    # --- ポジション数チェック（銘柄別の上限） ---
    can_open, pos_msg = can_open_new_position(symbol, data.positions)
    if not can_open:
        logger.info(f"🚫 {symbol}: {pos_msg}")
        return {"action": "NO_TRADE", "comment": pos_msg, "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": "", "news_status": ""}

    # 2. 閾値（エントリー条件）の決定
    buy_thresh = current_settings["buy_thresh"]
    sell_thresh = current_settings["sell_thresh"]

    current_price = data.ask
    if symbol not in price_history: price_history[symbol] = []
    price_history[symbol].append(current_price)
    # 銘柄別のhistory_sizeを使用
    if len(price_history[symbol]) > symbol_history_size + 10: price_history[symbol].pop(0)
    history = price_history[symbol]

    if len(history) < symbol_history_size:
        return {"action": "NO_TRADE", "comment": f"Learning... ({len(history)}/{symbol_history_size})", "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": "", "news_status": ""}

    # --- 時間帯フィルター ---
    is_active_time, session_info = is_active_trading_time(symbol)
    if not is_active_time:
        logger.info(f"⏰ {symbol}: {session_info} - 取引時間外")
        return {"action": "NO_TRADE", "comment": session_info, "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": "", "news_status": ""}

    highest, lowest = find_high_low(history)
    price_range = highest - lowest
    position = (current_price - lowest) / price_range if price_range > 0 else 0.5
    slope, _ = linear_regression_channel(history)
    atr = calculate_atr(history, ATR_PERIOD) * atr_multiplier  # ATR倍率を適用
    rsi = calculate_rsi(history)  # RSI計算
    adx = calculate_adx(history)  # v8.0: ADX計算

    logger.info(f"Env: {symbol} | Pos={position:.2f} | Slope={slope:.5f} | RSI={rsi:.1f} | ADX={adx:.1f} | ATR={atr:.5f} | Session={session_info}")

    signal = "NO_TRADE"; comment = "Wait"; sl=0.0; tp=0.0; trend="None"
    mid_price = (highest + lowest) / 2

    # ============================================================
    # v10.5: デュアルモード対応エントリーフィルター
    # STABLE: 安定重視（M15/H1）- 低頻度・高勝率
    # AGGRESSIVE: 収益重視（M5）- 高頻度・アフィリエイト増
    # ============================================================
    entry_params = get_entry_params()  # TRADE_MODEに応じたパラメータ取得
    params = entry_params.get(symbol, entry_params["DEFAULT"])

    # --- レジーム検出（パラメータ自動調整） ---
    detected_regime = ""
    try:
        if symbol not in regime_detectors:
            regime_detectors[symbol] = RegimeDetector(window_size=50)

        detector = regime_detectors[symbol]

        # Lazy-fit: 初回のみ学習（price historyからcandle dictsを構築）
        if not detector.is_fitted and len(history) >= 100:
            candles = [{"open": p, "high": p, "low": p, "close": p} for p in history]
            detector.fit(candles)
            logger.info(f"🧠 RegimeDetector fitted for {symbol} ({len(history)} candles)")

        if detector.is_fitted and len(history) >= 50:
            candles = [{"open": p, "high": p, "low": p, "close": p} for p in history]
            regime_result = detector.detect(candles)
            detected_regime = regime_result.regime  # "TRENDING", "RANGING", "VOLATILE"
            params = detector.get_regime_params(regime_result.regime, params)
            logger.info(f"🧠 Regime: {symbol} = {regime_result.regime} (conf={regime_result.confidence:.2f}) → params adjusted")
    except Exception as e:
        logger.warning(f"RegimeDetector error for {symbol}: {e}")

    # --- v10.0: ADXフィルター（厳格化） ---
    adx_threshold = params["adx_threshold"]
    if adx < adx_threshold:
        logger.info(f"📉 v10: {symbol}: ADX={adx:.1f} < {adx_threshold} → 弱トレンド、見送り")
        return {"action": "NO_TRADE", "comment": f"v10_ADX{adx:.0f}", "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": detected_regime, "news_status": news_status_str}

    # --- v10.0: RSI極端値フィルター（全シンボル共通） ---
    if params["rsi_extreme_avoid"] and (rsi < 25 or rsi > 75):
        logger.info(f"📉 v10: {symbol}: RSI={rsi:.1f} → 極端値、見送り")
        return {"action": "NO_TRADE", "comment": f"v10_RSI{rsi:.0f}", "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": detected_regime, "news_status": news_status_str}

    # --- v10.0: Slopeフィルター（厳格化） ---
    slope_threshold = params["slope_threshold"]
    abs_slope = abs(slope)
    if abs_slope < slope_threshold:
        logger.info(f"📉 v10: {symbol}: Slope={slope:.5f} → トレンド不明確、見送り")
        return {"action": "NO_TRADE", "comment": f"v10_Slope", "sl": 0.0, "tp": 0.0, "used_persona": use_persona, "regime": detected_regime, "news_status": news_status_str}

    # --- v10.0: エントリー判断（厳格版） ---
    # BUY条件: 強い上昇トレンド + 深い安値圏 + RSI過熱なし
    buy_position_thresh = params["buy_position"]
    rsi_buy_max = params["rsi_buy_max"]

    if slope > slope_threshold and position < buy_position_thresh and rsi < rsi_buy_max:
        signal = "BUY"
        trend = f"v10_Up_ADX{adx:.0f}"
        # v10.0: 高勝率型SL/TP（TP狭め、SL広め）
        tp = current_price + (atr * params["tp_mult"])
        sl = current_price - (atr * params["sl_mult"])
        logger.info(f"✅ v10 BUY: Slope={slope:.5f} Pos={position:.2f} RSI={rsi:.0f}")

    # SELL条件: 強い下降トレンド + 深い高値圏 + RSI過熱なし
    sell_position_thresh = params["sell_position"]
    rsi_sell_min = params["rsi_sell_min"]

    if slope < -slope_threshold and position > sell_position_thresh and rsi > rsi_sell_min:
        signal = "SELL"
        trend = f"v10_Down_ADX{adx:.0f}"
        # v10.0: 高勝率型SL/TP（TP狭め、SL広め）
        tp = current_price - (atr * params["tp_mult"])
        sl = current_price + (atr * params["sl_mult"])
        logger.info(f"✅ v10 SELL: Slope={slope:.5f} Pos={position:.2f} RSI={rsi:.0f}")

    if signal != "NO_TRADE":
        comment = trend
        logger.info(f"✅ v10.0 Entry: {signal} | ADX={adx:.1f} | SL={sl:.5f} | TP={tp:.5f}")

    return {"action": signal, "sl": round(sl,5), "tp": round(tp,5), "comment": comment, "used_persona": use_persona, "regime": detected_regime, "news_status": news_status_str}

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    conn = sqlite3.connect(DATABASE_NAME); c = conn.cursor()
    try: c.execute("SELECT timestamp, symbol, action, comment, persona FROM trade_logs ORDER BY id DESC LIMIT 10"); logs = c.fetchall()
    except: logs = []
    conn.close()
    
    # Generate Rows
    log_rows = "".join([f"<tr><td>{l[0]}</td><td>{l[1]}</td><td>{l[2]}</td><td>{l[3]}</td><td>{l[4] if l[4] else 'Unknown'}</td></tr>" for l in logs])
    
    pos_rows = ""
    if current_positions:
        for p in current_positions:
            p_color = "red" if p.profit < 0 else "green"
            pos_rows += f"<tr><td>{p.ticket}</td><td>{p.symbol}</td><td>{p.type}</td><td>{p.vol}</td><td>{p.open}</td><td>{p.current}</td><td style='color:{p_color}; font-weight:bold;'>{p.profit}</td></tr>"
    else:
        pos_rows = "<tr><td colspan='7' style='text-align:center;'>No Open Positions</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI EA Ultimate Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; max-width: 950px; margin: 0 auto; padding: 20px; background-color: #f0f2f5; color: #333; }}
            .container {{ display: flex; flex-direction: column; gap: 20px; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
            h1 {{ text-align: center; color: #2c3e50; margin-bottom: 20px; }}
            
            /* Preset Buttons */
            .preset-container {{ display: flex; gap: 10px; margin-bottom: 20px; }}
            .preset-btn {{ flex: 1; padding: 15px; border: none; border-radius: 8px; cursor: pointer; color: white; font-weight: bold; font-size: 1.1em; transition: 0.2s; }}
            .preset-btn:hover {{ opacity: 0.9; transform: translateY(-2px); }}
            
            .btn-red {{ background: linear-gradient(135deg, #ff416c, #ff4b2b); box-shadow: 0 4px 10px rgba(255, 75, 43, 0.3); }}
            .btn-blue {{ background: linear-gradient(135deg, #3498db, #2c3e50); box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3); }}
            .btn-green {{ background: linear-gradient(135deg, #56ab2f, #a8e063); box-shadow: 0 4px 10px rgba(86, 171, 47, 0.3); }}
            
            .form-group {{ margin-bottom: 20px; }}
            .form-group label {{ display: block; margin-bottom: 8px; font-weight: bold; color: #34495e; }}
            .form-group input, .form-group select {{ width: 100%; padding: 12px; border: 2px solid #ecf0f1; border-radius: 8px; font-size: 1.1em; transition: 0.3s; }}
            .form-group input:focus {{ border-color: #3498db; outline: none; }}
            
            .apply-btn {{ width: 100%; padding: 15px; background-color: #2c3e50; color: white; border: none; border-radius: 8px; font-size: 1.2em; font-weight: bold; cursor: pointer; margin-top: 10px; transition: 0.2s; }}
            .apply-btn:hover {{ background-color: #34495e; }}
            
            table {{ width: 100%; border-collapse: collapse; font-size: 0.95em; }}
            th, td {{ border-bottom: 1px solid #eee; padding: 15px 10px; text-align: left; }}
            th {{ color: #7f8c8d; background-color: #f8f9fa; font-weight: 600; }}
            
            .status-box {{ background-color: #e8f4fd; padding: 20px; border-radius: 10px; border-left: 6px solid #3498db; margin-bottom: 25px; }}
            .recommendation {{ background-color: #fff9db; padding: 15px; border-radius: 8px; border: 1px solid #ffeaa7; color: #d35400; font-weight: bold; margin-top: 10px; display: none; }}
        </style>
        <script>
            // プリセット定義
            const presets = {{
                "Aggressive": {{ "buy": 0.9, "sell": 0.1, "desc": "【ブレイクアウト狙い】高値でも買い、安値でも売る。ホームラン狙いの設定です。" }},
                "Balanced":   {{ "buy": 0.6, "sell": 0.4, "desc": "【王道バランス】押し目と戻り目を狙う、最も推奨される設定です。" }},
                "Conservative": {{ "buy": 0.3, "sell": 0.7, "desc": "【堅実防御】深い押し目まで待ち、リスクを極限まで減らす設定です。" }}
            }};

            function loadPreset(mode) {{
                document.getElementById('buy_input').value = presets[mode].buy;
                document.getElementById('sell_input').value = presets[mode].sell;
                document.getElementById('persona_select').value = mode;
                document.getElementById('desc_text').innerText = presets[mode].desc;
                updateRecommendation();
            }}

            // スマートアドバイザー
            function updateRecommendation() {{
                let buy = parseFloat(document.getElementById('buy_input').value);
                let sell = parseFloat(document.getElementById('sell_input').value);
                let msgDiv = document.getElementById('recommendation_msg');
                let modeSelect = document.getElementById('persona_select');
                let msg = "";
                let show = false;

                // 単純なアドバイスロジック
                if (buy >= 0.8 || sell <= 0.2) {{
                    msg = "💡 アドバイス: 特攻設定です。決済モードは「Aggressive」が推奨されます。";
                    show = true;
                }} else if (buy <= 0.4 || sell >= 0.7) {{
                    msg = "💡 アドバイス: 慎重設定です。決済モードは「Conservative」が推奨されます。";
                    show = true;
                }} else {{
                    msg = "💡 アドバイス: バランスの良い設定です。決済モードは「Balanced」が適しています。";
                    show = true;
                }}
                
                msgDiv.innerText = msg;
                msgDiv.style.display = show ? "block" : "none";
            }}
            
            window.onload = updateRecommendation;
        </script>
    </head>
    <body>
        <h1>🧬 AI Server: Ultimate Custom</h1>
        <div class="container">
            <div class="card">
                <h2>💰 Live Positions</h2>
                <table>
                    <thead>
                        <tr><th>Ticket</th><th>Symbol</th><th>Type</th><th>Vol</th><th>Open</th><th>Current</th><th>P&L</th></tr>
                    </thead>
                    <tbody>
                        {pos_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>⚙️ Strategy Tuner</h2>
                <div class="status-box">
                    <div style="font-size:0.9em; color:#7f8c8d; margin-bottom:5px;">ACTIVE SETTINGS</div>
                    <div style="font-size:1.4em;">
                        Mode: <b>{current_settings['persona_name']}</b><br>
                        Buy Line: <b>{current_settings['buy_thresh']}</b> / Sell Line: <b>{current_settings['sell_thresh']}</b>
                    </div>
                </div>

                <p style="font-weight:bold;">1. プリセットから一括設定 (Click to Load)</p>
                <div class="preset-container">
                    <button type="button" class="preset-btn btn-red" onclick="loadPreset('Aggressive')">🔥 Aggressive</button>
                    <button type="button" class="preset-btn btn-blue" onclick="loadPreset('Balanced')">⚖️ Balanced</button>
                    <button type="button" class="preset-btn btn-green" onclick="loadPreset('Conservative')">🛡️ Conservative</button>
                </div>

                <p style="font-weight:bold; margin-top:20px;">2. 詳細チューニング (Fine Tuning)</p>
                <div style="background:#f8f9fa; padding:15px; border-radius:8px; margin-bottom:15px; color:#555;" id="desc_text">
                    プリセットボタンを押すか、数値を直接入力してください。
                </div>

                <form action="/update_settings" method="post">
                    <div class="form-group">
                        <label>エントリー数値設定 (Entry Thresholds)</label>
                        <div style="display:flex; gap:10px;">
                            <div style="flex:1;">
                                <label style="font-size:0.8em;">買い判定 (0.0~1.0)</label>
                                <input type="number" step="0.05" name="buy_thresh" id="buy_input" 
                                       value="{current_settings['buy_thresh']}" oninput="updateRecommendation()">
                            </div>
                            <div style="flex:1;">
                                <label style="font-size:0.8em;">売り判定 (0.0~1.0)</label>
                                <input type="number" step="0.05" name="sell_thresh" id="sell_input" 
                                       value="{current_settings['sell_thresh']}" oninput="updateRecommendation()">
                            </div>
                        </div>
                        <div id="recommendation_msg" class="recommendation"></div>
                    </div>

                    <div class="form-group">
                        <label>決済モード選択 (Exit Strategy)</label>
                        <select name="persona_name" id="persona_select">
                            <option value="Aggressive">🔥 Aggressive (Profit Focus)</option>
                            <option value="Balanced" selected>⚖️ Balanced (Stability)</option>
                            <option value="Conservative">🛡️ Conservative (Safety)</option>
                        </select>
                        <div style="font-size:0.85em; color:#7f8c8d; margin-top:5px;">
                            ※ モードによって損切り・利確の計算式が変わります。
                        </div>
                    </div>

                    <button type="submit" class="apply-btn">設定をサーバーに適用 (Apply)</button>
                </form>
            </div>

            <div class="card">
                <h2>📊 Live Trade History <a href="/dashboard" style="font-size:0.8em; margin-left:10px; text-decoration:none;">🔄</a></h2>
                <table>
                    <thead>
                        <tr><th>Time</th><th>Symbol</th><th>Action</th><th>Comment</th><th>Mode</th></tr>
                    </thead>
                    <tbody>
                        {log_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.post("/update_settings", response_class=RedirectResponse)
async def update_settings(
    persona_name: str = Form(...), buy_thresh: float = Form(...), sell_thresh: float = Form(...)
):
    global current_settings
    current_settings["persona_name"] = persona_name
    current_settings["buy_thresh"] = buy_thresh
    current_settings["sell_thresh"] = sell_thresh
    logger.info(f"🔄 Settings Updated: {current_settings}")
    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/history")
def update_history(data: HistoryData):
    if data.account_id not in ALLOWED_ACCOUNTS: return {"status": "error"}
    price_history[data.symbol] = data.prices
    logger.info(f"Loaded History: {len(data.prices)} bars")
    return {"status": "ok"}

# --- トレード結果報告エンドポイント（クールダウン管理用） ---
class TradeResult(BaseModel):
    account_id: int
    symbol: str
    is_loss: bool  # True=損失, False=利益

@app.post("/trade_result")
def report_trade_result(data: TradeResult):
    """EAからトレード結果を報告（クールダウン管理用）"""
    if data.account_id not in ALLOWED_ACCOUNTS:
        return {"status": "error", "message": "Invalid account"}

    record_trade_result(data.symbol, data.is_loss)
    config = get_symbol_config(data.symbol)
    state = cooldown_state.get(data.symbol, {})

    return {
        "status": "ok",
        "symbol": data.symbol,
        "consecutive_losses": state.get("consecutive_losses", 0),
        "cooldown_minutes": config["cooldown_minutes"] if state.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES else 0
    }

# ============================================================
# v7.0: 市場分析エンドポイント（AIは分析のみ）
# ============================================================
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_market(data: AnalyzeRequest):
    """AIに市場分析を依頼（判断はせず、コンテキストのみ提供）"""
    if data.account_id not in ALLOWED_ACCOUNTS:
        return {
            "symbol": data.symbol,
            "trend": "range", "strength": 5, "volatility": "medium",
            "risk_level": "medium", "recommendation": "認証エラー",
            "cached": False, "server_time": str(datetime.datetime.now())
        }

    symbol = data.symbol

    # 価格履歴を取得（リクエストから or キャッシュから）
    if data.prices and len(data.prices) > 20:
        history = data.prices
        price_history[symbol] = data.prices  # キャッシュも更新
    else:
        history = price_history.get(symbol, [])

    if len(history) < 20:
        return {
            "symbol": symbol,
            "trend": "range", "strength": 5, "volatility": "medium",
            "risk_level": "medium", "recommendation": "データ不足",
            "cached": False, "server_time": str(datetime.datetime.now())
        }

    # 指標計算
    rsi = calculate_rsi(history)
    slope, _ = linear_regression_channel(history)
    atr = calculate_atr(history, 14)

    # AI分析を実行
    analysis = ask_ai_market_analysis(symbol, history, rsi, slope, atr)

    logger.info(f"📊 Market Analysis: {symbol} | Trend={analysis['trend']} | Strength={analysis['strength']} | Risk={analysis['risk_level']}")

    return {
        "symbol": symbol,
        "trend": analysis["trend"],
        "strength": analysis["strength"],
        "volatility": analysis["volatility"],
        "risk_level": analysis["risk_level"],
        "recommendation": analysis["recommendation"],
        "cached": analysis.get("cached", False),
        "server_time": str(datetime.datetime.now())
    }


# ============================================================
# v9.0: 決済判断エンドポイント（トレーリング、BE、分割決済対応）
# ============================================================
@app.post("/check_exit", response_model=ExitCheckResponse)
def check_exit(data: ExitCheckRequest):
    """
    v9.0: 高度な決済判断（トレーリングストップ、ブレークイーブン、分割決済）
    """
    if data.account_id not in ALLOWED_ACCOUNTS:
        return {"action": "HOLD", "reason": "License Invalid", "server_time": str(datetime.datetime.now())}

    symbol = data.symbol

    # 保有時間を計算（分）
    open_time = datetime.datetime.fromtimestamp(data.open_time)
    holding_minutes = int((datetime.datetime.now() - open_time).total_seconds() / 60)

    # v9.0: リクエストからpricesが渡された場合はそれを優先
    if data.prices and len(data.prices) >= 20:
        history = data.prices
        price_history[symbol] = data.prices  # キャッシュも更新
    else:
        history = price_history.get(symbol, [])

    if len(history) < 20:
        # データ不足時はHOLD
        return {"action": "HOLD", "reason": "DataInsufficient", "server_time": str(datetime.datetime.now())}

    # v9.0: テクニカル指標を計算
    rsi = calculate_rsi(history)
    slope, _ = linear_regression_channel(history)
    adx = calculate_adx(history)

    # v9.0: 高度な決済判断（トレーリング、BE、分割決済）
    result = rule_based_exit_decision_v9(
        symbol=symbol,
        position_type=data.position_type,
        profit=data.profit,
        holding_minutes=holding_minutes,
        rsi=rsi,
        slope=slope,
        adx=adx,
        open_price=data.open_price,
        current_price=data.current_price,
        current_sl=data.sl,
        max_profit_seen=data.max_profit_seen,
        partial_closed=data.partial_closed,
        prices=history
    )

    # ログ出力（v9.0情報を含める）
    logger.info(f"🎯 Exit Check [v9.0]: {symbol} | {data.position_type} | Profit={data.profit:.2f} | Hold={holding_minutes}min")
    logger.info(f"   Indicators: ADX={adx:.1f} | RSI={rsi:.1f} | Slope={slope:.6f}")
    logger.info(f"   Decision: {result['action']} | Reason: {result['reason']}")
    if result['new_sl'] > 0:
        logger.info(f"   New SL: {result['new_sl']:.2f}")
    if result['partial_ratio'] > 0:
        logger.info(f"   Partial Ratio: {result['partial_ratio']*100:.0f}%")

    return {
        "action": result["action"],
        "reason": result["reason"],
        "server_time": str(datetime.datetime.now()),
        "new_sl": result["new_sl"],
        "partial_close": result["action"] == "PARTIAL_CLOSE",
        "partial_ratio": result["partial_ratio"]
    }

# ============================================================
# Report endpoint: AI-generated optimization report (Japanese)
# ============================================================
@app.get("/report/{symbol}")
def get_optimization_report(symbol: str):
    """Generate Japanese optimization report for a symbol."""
    history_path = os.path.join(CONFIG_PARAMS_DIR, "optimization_history.json")
    if not os.path.exists(history_path):
        return {"error": "No optimization history found"}

    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load history: {e}"}

    # Find the latest run for this symbol
    runs = [r for r in history if r.get("symbol") == symbol]
    if not runs:
        return {"error": f"No optimization runs found for {symbol}"}

    latest_run = runs[-1]

    generator = ReportGenerator(api_key=OPENAI_API_KEY)
    report = generator.generate(latest_run)

    return {"symbol": symbol, "report": report}


@app.post("/signal", response_model=TradeSignal)
def get_signal(data: MarketData):
    if data.account_id not in ALLOWED_ACCOUNTS: return {"action": "NO_TRADE", "sl_price": 0, "tp_price": 0, "comment": "License Invalid", "server_time": str(datetime.datetime.now())}
    result = analyze_market_logic(data)
    save_log(data, result, result["used_persona"])
    return {"action": result["action"], "sl_price": result["sl"], "tp_price": result["tp"], "comment": result["comment"], "server_time": str(datetime.datetime.now()), "regime": result.get("regime", ""), "news_status": result.get("news_status", "")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)