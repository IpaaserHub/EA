//+------------------------------------------------------------------+
//|                                          AI_Trading_Bot_UI.mq5    |
//|           AI Driven Expert Advisor (Ver 7.00 AI Exit Decision)    |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, AI Project"
#property version   "7.00"
#property strict

//--- 性格モード
enum ENUM_AI_PERSONA { Aggressive, Balanced, Conservative };
ENUM_AI_PERSONA CurrentPersona = Balanced; 

enum ENUM_SLTP_MODE
  {
   AI_Auto_Logic,    
   User_Fixed_Points 
  };

//--- インプットパラメータ
input group "--- Server Connection ---"
input string   ServerURL         = "http://127.0.0.1:8000"; 
input int      MagicNumber       = 123456;      

input group "--- Risk Management ---"
input double   FixedLot          = 0.01;        
input double   MaxDailyLoss      = 5000;        
input int      MaxConsecutiveLoss = 3;          

input group "--- User Custom Settings ---"
input ENUM_SLTP_MODE SL_TP_Mode  = AI_Auto_Logic; 
input int      User_TP_Points    = 500;         
input int      User_SL_Points    = 300;         

input group "--- Trailing Stop ---"
input bool     UseTrailingStop   = true;
input int      TrailingStart     = 200;
input int      TrailingStep      = 50;

input group "--- AI Exit Check (Phase 2.3) ---"
input bool     UseAIExitCheck    = true;        // AI決済判断を使用
input int      ExitCheckInterval = 60;          // チェック間隔（秒）

input group "--- System Settings ---"
input int      HistoryBars       = 100;         

//--- UIオブジェクト名（定数的に使用）
string BtnAggressive = "BtnAggressive";
string BtnBalanced   = "BtnBalanced";
string BtnConservative = "BtnConservative";
string StatusLabel = "StatusLabel";
string MsgLabel = "MsgLabel";
string BtnPanic = "BtnPanic";
string BtnExport = "BtnExport"; // CSV出力ボタン
string RegimeLabel = "RegimeLabel"; // v7.0: レジーム表示
string NewsLabel = "NewsLabel";     // v7.0: ニュース状態表示

//--- グローバル変数（ここにday_of_yearがあることを確認！）
datetime last_bar_time = 0;
int      consecutive_losses = 0;
double   initial_balance_day = 0;
int      day_of_year = 0;          // ★日付管理用変数
bool     is_stopped_today = false;
bool     is_history_sent = false;
datetime last_exit_check_time = 0; // AI決済チェック用タイムスタンプ

//--- v7.0: Per-position state tracking (parallel arrays)
#define MAX_TRACKED_POSITIONS 20
ulong  tracked_tickets[MAX_TRACKED_POSITIONS];
double tracked_max_profit[MAX_TRACKED_POSITIONS];
bool   tracked_partial_closed[MAX_TRACKED_POSITIONS];
int    tracked_count = 0;

//--- 関数プロトタイプ
void CheckNewDay();
bool CheckDailyDrawdown();
void ManageTrailingStop();
string GetSignalFromServer();
void SendHistoryToServer();
void ProcessSignal(string json);
void TradeOpen(ENUM_ORDER_TYPE type, double sl, double tp, string comment);
void ModifySL(ulong ticket, double sl);
string ParseJsonString(string json, string key);
double ParseJsonDouble(string json, string key);
ENUM_ORDER_TYPE_FILLING GetFillingMode();
void CreateButton(string name, string text, int x, int y, int w, int h, color bg, color border, int corner, int fontsize);
void CreateLabel(string name, string text, int x, int y, int fontsize, color col);
void UpdateButtonState();
void CloseAllPositions();
void ExportHistoryToCSV();
void ReportTradeResult(string symbol, bool is_loss); // クールダウン用結果報告
void CheckAIExitSignals();  // Phase 2.3: AI決済判断
bool ClosePositionByTicket(ulong ticket); // 個別ポジション決済
bool PartialCloseByTicket(ulong ticket, double ratio); // v7.0: 分割決済
int  FindTrackedIndex(ulong ticket);          // v7.0: 追跡インデックス検索
void AddTrackedPosition(ulong ticket);        // v7.0: 追跡追加
void RemoveTrackedPosition(ulong ticket);     // v7.0: 追跡削除
void CleanupTrackedPositions();               // v7.0: 追跡クリーンアップ

//+------------------------------------------------------------------+
//| 初期化                                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("AI EA (Ver 7.00 AI Exit Decision) Initialized.");
   initial_balance_day = AccountInfoDouble(ACCOUNT_BALANCE);
   
   // 日付初期化
   MqlDateTime dt;
   TimeCurrent(dt);
   day_of_year = dt.day_of_year;

   // UI作成
   // 性格ボタン（白枠）
   CreateButton("BtnAggressive", "🔥 Aggressive", 10, 30, 90, 25, C'231,76,60', clrWhite, CORNER_LEFT_UPPER, 9);
   CreateButton("BtnBalanced", "⚖️ Balanced", 110, 30, 90, 25, C'52,152,219', clrWhite, CORNER_LEFT_UPPER, 9);
   CreateButton("BtnConservative", "🛡️ Conservative", 210, 30, 100, 25, C'39,174,96', clrWhite, CORNER_LEFT_UPPER, 9);
   CreateLabel("StatusLabel", "Current Mode: Balanced", 10, 10, 9, clrWhite);

   CreateLabel("MsgLabel", "AI: Waiting for signal...", 10, 70, 12, clrYellow);

   // 【デザイン修正】緊急停止ボタン
   // 正方形(40x40)、赤背景、黄色枠、⚠マーク
   CreateButton("BtnPanic", "⚠", 10, 30, 40, 40, clrRed, clrYellow, CORNER_RIGHT_UPPER, 20);
   
   // CSV出力ボタン（緊急停止ボタンの下）
   CreateButton("BtnExport", "📄 CSV", 10, 80, 60, 25, C'100,100,100', clrWhite, CORNER_RIGHT_UPPER, 8);

   // v7.0: レジーム・ニュースラベル
   CreateLabel(RegimeLabel, "Regime: ---", 10, 95, 9, clrWhite);
   CreateLabel(NewsLabel, "News: Clear", 10, 115, 9, C'46,204,113');

   UpdateButtonState();
   ChartRedraw();
   
   // ★新規追加: 1秒ごとのタイマーイベントを設定（土日や値動きがない時でも通信するため）
   EventSetTimer(1);
   
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer(); // タイマー破棄

   // UI削除
   ObjectDelete(0, "BtnAggressive");
   ObjectDelete(0, "BtnBalanced");
   ObjectDelete(0, "BtnConservative");
   ObjectDelete(0, "StatusLabel");
   ObjectDelete(0, "MsgLabel");
   ObjectDelete(0, "BtnPanic");
   ObjectDelete(0, "BtnExport");
   ObjectDelete(0, RegimeLabel);
   ObjectDelete(0, NewsLabel);
   Print("AI EA Stopped.");
  }

//+------------------------------------------------------------------+
//| トレード結果検知（クールダウン用）                                   |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   // 約定追加イベントのみ処理
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      if(HistoryDealSelect(trans.deal))
      {
         long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         long magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);

         // 決済（OUT）かつ自分のマジックナンバーの場合のみ処理
         if(entry == DEAL_ENTRY_OUT && magic == MagicNumber)
         {
            string symbol = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
            double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
            double swap = HistoryDealGetDouble(trans.deal, DEAL_SWAP);
            double total_profit = profit + swap;
            bool is_loss = (total_profit < 0);

            Print("📊 Trade Closed: ", symbol, " | P/L: ", total_profit, " | Loss: ", is_loss);

            // サーバーに結果を報告
            ReportTradeResult(symbol, is_loss);

            // ローカルの連敗カウンタも更新
            if(is_loss)
               consecutive_losses++;
            else
               consecutive_losses = 0;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| タイマーイベント（土日対応）                                        |
//+------------------------------------------------------------------+
void OnTimer()
{
   // タイマーでも通常のメインループ(OnTick)と同じ処理を行う
   OnTick();
}

//+------------------------------------------------------------------+
//| チャートイベント                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(id == CHARTEVENT_OBJECT_CLICK)
     {
      if(sparam == "BtnAggressive") CurrentPersona = Aggressive;
      else if(sparam == "BtnBalanced") CurrentPersona = Balanced;
      else if(sparam == "BtnConservative") CurrentPersona = Conservative;
      else if(sparam == "BtnPanic")
      {
         // メッセージボックス
         if(MessageBox("【警告】\n緊急停止しますか？(全決済・EA停止)\nEmergency Stop?", 
                       "PANIC", MB_YESNO|MB_ICONSTOP) == IDYES)
         {
            Print("🚨 PANIC BUTTON PRESSED! CLOSING ALL POSITIONS...");
            CloseAllPositions();
            ExpertRemove(); 
         }
      }
      else if(sparam == "BtnExport")
      {
         ExportHistoryToCSV();
      }
      UpdateButtonState();
      ChartRedraw();
     }
  }

//+------------------------------------------------------------------+
//| メインループ                                                      |
//+------------------------------------------------------------------+
void OnTick()
  {
   if (!is_history_sent) {
      SendHistoryToServer();
      is_history_sent = true;
   }

   CheckNewDay();

   if(is_stopped_today) {
      ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, "AI: Stopped (Daily Loss Limit)");
      return;
   }

   if (CheckDailyDrawdown()) {
      is_stopped_today = true;
      return;
   }

   if(UseTrailingStop) ManageTrailingStop();

   // Phase 2.3: AI決済判断（ポジション保有中、指定間隔で実行）
   if(UseAIExitCheck && PositionsTotal() > 0)
   {
      datetime now = TimeCurrent();
      if(now - last_exit_check_time >= ExitCheckInterval)
      {
         last_exit_check_time = now;
         CheckAIExitSignals();
      }
   }

   datetime current_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(last_bar_time == current_time) return;

   if(PositionsTotal() > 0) {
       ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, "AI: Monitoring Positions...");
       // return; // 【修正】ポジション保有中もサーバーと通信して状況を報告させるため、ここのreturnを削除
   }

   last_bar_time = current_time;

   string signal = GetSignalFromServer();
   ProcessSignal(signal);
  }

//+------------------------------------------------------------------+
//| 緊急全決済関数                                                     |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
      {
         MqlTradeRequest request;
         MqlTradeResult  result;
         ZeroMemory(request);
         ZeroMemory(result);

         request.action   = TRADE_ACTION_DEAL;
         request.position = ticket;
         request.symbol   = PositionGetString(POSITION_SYMBOL);
         request.volume   = PositionGetDouble(POSITION_VOLUME);
         request.deviation= 100;
         request.type_filling = GetFillingMode();
         
         long type = PositionGetInteger(POSITION_TYPE);
         if(type == POSITION_TYPE_BUY) {
            request.type = ORDER_TYPE_SELL;
            request.price= SymbolInfoDouble(request.symbol, SYMBOL_BID);
         } else {
            request.type = ORDER_TYPE_BUY;
            request.price= SymbolInfoDouble(request.symbol, SYMBOL_ASK);
         }
         
         if(!OrderSend(request, result)) {
             Print("Emergency Close Error: ", GetLastError());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| サーバー通信                                                      |
//+------------------------------------------------------------------+
// --- 新規追加: 保有ポジションをJSON配列化する関数 ---
string GetPositionsJson()
{
   string json = "[";
   int total = PositionsTotal();
   bool first = true;
   
   for(int i=0; i<total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      // 【修正】マジックナンバー制限を撤廃（全てのポジションを表示する）
      // if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
      {
         if(!first) json += ",";
         
         string symbol = PositionGetString(POSITION_SYMBOL);
         long type = PositionGetInteger(POSITION_TYPE);
         double vol = PositionGetDouble(POSITION_VOLUME);
         double open = PositionGetDouble(POSITION_PRICE_OPEN);
         double sl = PositionGetDouble(POSITION_SL);
         double tp = PositionGetDouble(POSITION_TP);
         double current = PositionGetDouble(POSITION_PRICE_CURRENT);
         double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
         
         string type_str = (type==POSITION_TYPE_BUY) ? "BUY" : "SELL";
         
         string item = StringFormat(
            "{\"ticket\":%d, \"symbol\":\"%s\", \"type\":\"%s\", \"vol\":%.2f, \"open\":%.5f, \"sl\":%.5f, \"tp\":%.5f, \"current\":%.5f, \"profit\":%.2f}",
            ticket, symbol, type_str, vol, open, sl, tp, current, profit
         );
         
         json += item;
         first = false;
      }
   }
   json += "]";
   return json;
}

string GetSignalFromServer()
{
   double daily_profit = AccountInfoDouble(ACCOUNT_EQUITY) - initial_balance_day;
   
   string persona_str = "Balanced";
   if(CurrentPersona == Aggressive) persona_str = "Aggressive";
   else if(CurrentPersona == Conservative) persona_str = "Conservative";

   string positions_json = GetPositionsJson();

   string json_request = StringFormat(
      "{\"account_id\": %s, \"symbol\": \"%s\", \"bid\": %.5f, \"ask\": %.5f, \"bar_time\": %s, \"equity\": %.2f, \"daily_profit\": %.2f, \"persona\": \"%s\", \"positions\": %s}",
      IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)),
      _Symbol,
      SymbolInfoDouble(_Symbol, SYMBOL_BID),
      SymbolInfoDouble(_Symbol, SYMBOL_ASK),
      IntegerToString((long)last_bar_time),
      AccountInfoDouble(ACCOUNT_EQUITY),
      daily_profit,
      persona_str,
      positions_json
   );

   char post_data[];
   int len = StringLen(json_request);
   StringToCharArray(json_request, post_data, 0, len);
   char result_data[];
   string result_headers;
   string headers = "Content-Type: application/json\r\n";
   
   // タイムアウトを 5000 -> 10000 に延長
   int res_code = WebRequest("POST", ServerURL + "/signal", headers, 10000, post_data, result_data, result_headers);
   
   if(res_code == 200) return CharArrayToString(result_data);
   else {
      ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, "AI: Connection Error " + IntegerToString(res_code));
      return "";
   }
}

//+------------------------------------------------------------------+
//| シグナル処理                                                      |
//+------------------------------------------------------------------+
void ProcessSignal(string json)
{
   if(json == "") return;

   string action = ParseJsonString(json, "action");
   string comment = ParseJsonString(json, "comment");

   string display_msg = "AI: " + action + " (" + comment + ")";
   ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, display_msg);

   // v7.0: Update regime label
   string regime = ParseJsonString(json, "regime");
   if(regime != "")
   {
      ObjectSetString(0, RegimeLabel, OBJPROP_TEXT, "Regime: " + regime);
      if(regime == "TRENDING")
         ObjectSetInteger(0, RegimeLabel, OBJPROP_COLOR, C'46,204,113');  // green
      else if(regime == "RANGING")
         ObjectSetInteger(0, RegimeLabel, OBJPROP_COLOR, C'241,196,15');  // yellow
      else if(regime == "VOLATILE")
         ObjectSetInteger(0, RegimeLabel, OBJPROP_COLOR, C'231,76,60');   // red
      else
         ObjectSetInteger(0, RegimeLabel, OBJPROP_COLOR, clrWhite);
   }
   else
   {
      ObjectSetString(0, RegimeLabel, OBJPROP_TEXT, "Regime: ---");
      ObjectSetInteger(0, RegimeLabel, OBJPROP_COLOR, clrWhite);
   }

   // v7.0: Update news label
   string news_status = ParseJsonString(json, "news_status");
   if(news_status != "")
   {
      ObjectSetString(0, NewsLabel, OBJPROP_TEXT, "News: " + news_status);
      ObjectSetInteger(0, NewsLabel, OBJPROP_COLOR, C'231,76,60');  // red
   }
   else
   {
      ObjectSetString(0, NewsLabel, OBJPROP_TEXT, "News: Clear");
      ObjectSetInteger(0, NewsLabel, OBJPROP_COLOR, C'46,204,113');  // green
   }

   ChartRedraw();

   double server_sl = ParseJsonDouble(json, "sl"); 
   if(server_sl == 0) server_sl = ParseJsonDouble(json, "sl_price");
   double server_tp = ParseJsonDouble(json, "tp");
   if(server_tp == 0) server_tp = ParseJsonDouble(json, "tp_price");

   if (action == "NO_TRADE") return;
   if (consecutive_losses >= MaxConsecutiveLoss) return;

   double final_sl = 0;
   double final_tp = 0;
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   if (SL_TP_Mode == AI_Auto_Logic) {
       final_sl = server_sl;
       final_tp = server_tp;
   } 
   else if (SL_TP_Mode == User_Fixed_Points) {
       double current_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
       double current_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
       if (action == "BUY") {
           final_sl = current_ask - (User_SL_Points * point);
           final_tp = current_ask + (User_TP_Points * point);
       } else if (action == "SELL") {
           final_sl = current_bid + (User_SL_Points * point);
           final_tp = current_bid - (User_TP_Points * point);
       }
       comment = comment + "_UserFixed";
   }

   if(action == "BUY") TradeOpen(ORDER_TYPE_BUY, final_sl, final_tp, comment);
   else if(action == "SELL") TradeOpen(ORDER_TYPE_SELL, final_sl, final_tp, comment);
}

//+------------------------------------------------------------------+
//| 注文実行                                                          |
//+------------------------------------------------------------------+
void TradeOpen(ENUM_ORDER_TYPE type, double sl, double tp, string comment)
{
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = _Symbol;
   
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double volume = FixedLot;
   
   if(volume < min_lot) volume = min_lot;
   volume = min_lot + MathFloor((volume - min_lot)/step_lot) * step_lot;

   request.volume    = volume;
   request.type      = type;
   request.price     = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl        = NormalizeDouble(sl, _Digits);
   request.tp        = NormalizeDouble(tp, _Digits);
   request.deviation = 20; 
   request.magic     = MagicNumber;
   request.comment   = comment;
   
   request.type_filling = GetFillingMode();

   if(OrderSend(request, result)) {
       Print("Order Opened: ", comment, " | Vol:", volume);
       ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, "AI: Order Opened! " + comment);
   } else {
       Print("Order Error: ", GetLastError(), " | RetCode:", result.retcode);
       ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, "AI: Order Error " + IntegerToString(GetLastError()));
   }
}

//+------------------------------------------------------------------+
//| 注文方式判定                                                      |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
   int filling = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC) return ORDER_FILLING_IOC;
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK) return ORDER_FILLING_FOK;
   return ORDER_FILLING_RETURN; 
}

//+------------------------------------------------------------------+
//| UI関連関数                                                        |
//+------------------------------------------------------------------+
void UpdateButtonState()
{
   bool agg = (CurrentPersona == Aggressive);
   bool bal = (CurrentPersona == Balanced);
   bool con = (CurrentPersona == Conservative);

   ObjectSetInteger(0, "BtnAggressive", OBJPROP_STATE, agg);
   ObjectSetInteger(0, "BtnBalanced", OBJPROP_STATE, bal);
   ObjectSetInteger(0, "BtnConservative", OBJPROP_STATE, con);
   
   string modeStr = "Balanced";
   if(agg) modeStr = "Aggressive";
   if(con) modeStr = "Conservative";
   
   ObjectSetString(0, "StatusLabel", OBJPROP_TEXT, "Current Mode: " + modeStr);
}

// 【修正】枠線の色を指定できるように変更
void CreateButton(string name, string text, int x, int y, int w, int h, color bg, color border, int corner, int fontsize)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, name, OBJPROP_BORDER_COLOR, border); // 枠線色
   ObjectSetInteger(0, name, OBJPROP_CORNER, corner);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontsize);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}

void CreateLabel(string name, string text, int x, int y, int fontsize, color col)
{
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontsize);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}

//+------------------------------------------------------------------+
//| トレイリングストップ                                               |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   long level_int = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double stops_level = (double)level_int * _Point;
   if(stops_level == 0) stops_level = 50 * _Point;

   for(int i=PositionsTotal()-1; i>=0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
         double current_sl = PositionGetDouble(POSITION_SL);
         double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
         double current_price = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
            if(current_price - open_price > TrailingStart * point) {
               double new_sl = current_price - TrailingStep * point;
               if (current_price - new_sl < stops_level) new_sl = current_price - stops_level - (10 * point);
               new_sl = NormalizeDouble(new_sl, _Digits);
               if(new_sl > current_sl && new_sl > open_price) ModifySL(ticket, new_sl);
            }
         }
         else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
            if(open_price - current_price > TrailingStart * point) {
               double new_sl = current_price + TrailingStep * point;
               if (new_sl - current_price < stops_level) new_sl = current_price + stops_level + (10 * point);
               new_sl = NormalizeDouble(new_sl, _Digits);
               if(new_sl < current_sl || current_sl == 0) ModifySL(ticket, new_sl);
            }
         }
      }
   }
}

void ModifySL(ulong ticket, double sl)
{
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);
   request.action   = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl       = NormalizeDouble(sl, _Digits);
   request.tp       = PositionGetDouble(POSITION_TP);
   if(!OrderSend(request, result)) Print("Order Modify Error: ", GetLastError());
}

//+------------------------------------------------------------------+
//| ユーティリティ                                                    |
//+------------------------------------------------------------------+
string ParseJsonString(string json, string key) {
   string searchKey = "\"" + key + "\":";
   int startPos = StringFind(json, searchKey);
   if(startPos == -1) return "";
   startPos += StringLen(searchKey);
   while(StringSubstr(json, startPos, 1) == " " || StringSubstr(json, startPos, 1) == "\"") startPos++;
   int endPos = startPos;
   while(StringSubstr(json, endPos, 1) != "\"" && StringSubstr(json, endPos, 1) != "," && StringSubstr(json, endPos, 1) != "}") endPos++;
   return StringSubstr(json, startPos, endPos - startPos);
}

double ParseJsonDouble(string json, string key) {
   string val = ParseJsonString(json, key);
   return StringToDouble(val);
}

void CheckNewDay(){ 
   MqlDateTime d; TimeCurrent(d); 
   if(d.day_of_year!=day_of_year){
      day_of_year=d.day_of_year;
      initial_balance_day=AccountInfoDouble(ACCOUNT_BALANCE);
      is_stopped_today=false;
   }
}

bool CheckDailyDrawdown(){
   double p=AccountInfoDouble(ACCOUNT_EQUITY)-initial_balance_day;
   if(p<-MaxDailyLoss) return true;
   return false;
}

void SendHistoryToServer() {
   Print("Syncing history data to AI Server...");
   double close_prices[];
   int copied = CopyClose(_Symbol, PERIOD_CURRENT, 1, HistoryBars, close_prices);
   
   if (copied <= 0) {
      Print("Error: Failed to copy history data.");
      return;
   }

   string prices_json = "[";
   for(int i=0; i<copied; i++) {
      prices_json += DoubleToString(close_prices[i], 5);
      if (i < copied - 1) prices_json += ",";
   }
   prices_json += "]";

   string json_request = StringFormat(
      "{\"account_id\": %s, \"symbol\": \"%s\", \"prices\": %s}",
      IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)),
      _Symbol,
      prices_json
   );

   char post_data[];
   int len = StringLen(json_request);
   StringToCharArray(json_request, post_data, 0, len);

   char result_data[];
   string result_headers;
   string headers = "Content-Type: application/json\r\n";
   
   int res = WebRequest("POST", ServerURL + "/history", headers, 10000, post_data, result_data, result_headers);
   
   if (res == 200) {
      Print("History Sync Success! Sent ", copied, " bars.");
      ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, "AI: Ready! History Loaded.");
   } else {
      Print("History Sync Failed. Code: ", res);
      ObjectSetString(0, "MsgLabel", OBJPROP_TEXT, "AI: Sync Failed (" + IntegerToString(res) + ")");
   }
}

//+------------------------------------------------------------------+
//| CSV出力機能 (改良版)                                              |
//+------------------------------------------------------------------+
void ExportHistoryToCSV()
{
   // ファイル名を作成 (AI_TradeHistory_MagicNo.csv)
   string filename = "AI_TradeHistory_" + IntegerToString(MagicNumber) + ".csv";
   
   // エラー状態をリセット
   ResetLastError();

   // CSVファイルを開く (書き込みモード, CSV形式, ANSIエンコード, 共有読み込み許可)
   int file_handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_SHARE_READ, ",");
   
   if(file_handle == INVALID_HANDLE) {
      int err = GetLastError();
      string errMsg = "CSV作成エラー (Error: " + IntegerToString(err) + ")";
      if(err == 5004) errMsg += "\nファイルがロックされています。Excel等を閉じてください。";
      MessageBox(errMsg, "Export Error", MB_ICONSTOP);
      return;
   }

   // ヘッダー行をMT5の「口座履歴」詳細に合わせて書き込み
   FileWrite(file_handle, "Time", "Ticket", "Symbol", "Type", "Direction", "Volume", "Price", "S/L", "T/P", "Commission", "Swap", "Profit", "Comment");

   // 全期間の履歴を選択
   if(!HistorySelect(0, TimeCurrent())) {
      FileClose(file_handle);
      MessageBox("履歴データの取得に失敗しました。", "Error", MB_ICONSTOP);
      return;
   }

   int total = HistoryDealsTotal();
   int count = 0;
   
   if (total == 0) {
      FileClose(file_handle);
      MessageBox("出力対象の履歴が0件です。\n(まだ取引が完了していない可能性があります)", "No History", MB_ICONWARNING);
      return;
   }
   
   // 履歴をループして書き込み
   for(int i = 0; i < total; i++) {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket > 0) {
         // プロパティ取得
         datetime time = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
         string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
         long type = HistoryDealGetInteger(ticket, DEAL_TYPE);
         long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY); // IN/OUT/INOUT

         double vol = HistoryDealGetDouble(ticket, DEAL_VOLUME);
         double price = HistoryDealGetDouble(ticket, DEAL_PRICE);
         double sl = HistoryDealGetDouble(ticket, DEAL_SL);
         double tp = HistoryDealGetDouble(ticket, DEAL_TP);
         
         double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
         double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         
         string comment = HistoryDealGetString(ticket, DEAL_COMMENT);

         // 列挙型を読みやすい文字列に変換
         string type_str = EnumToString((ENUM_DEAL_TYPE)type);
         string entry_str = EnumToString((ENUM_DEAL_ENTRY)entry);
         
         // 文字列の整形 (DEAL_TYPE_BUY -> BUY, DEAL_ENTRY_IN -> IN)
         StringReplace(type_str, "DEAL_TYPE_", "");
         StringReplace(entry_str, "DEAL_ENTRY_", "");

         // 行書き込み
         FileWrite(file_handle, TimeToString(time), ticket, symbol, type_str, entry_str, vol, price, sl, tp, commission, swap, profit, comment);
         count++;
      }
   }

   FileFlush(file_handle);
   FileClose(file_handle);
   
   // 完了メッセージ
   MessageBox("CSV出力完了！\n\n場所: MQL5/Files/" + filename + "\n件数: " + IntegerToString(count), "Export Success", MB_OK);
}

//+------------------------------------------------------------------+
//| サーバーにトレード結果を報告（クールダウン管理用）                    |
//+------------------------------------------------------------------+
void ReportTradeResult(string symbol, bool is_loss)
{
   string json_request = StringFormat(
      "{\"account_id\": %s, \"symbol\": \"%s\", \"is_loss\": %s}",
      IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)),
      symbol,
      is_loss ? "true" : "false"
   );

   char post_data[];
   int len = StringLen(json_request);
   StringToCharArray(json_request, post_data, 0, len);

   char result_data[];
   string result_headers;
   string headers = "Content-Type: application/json\r\n";

   int res_code = WebRequest("POST", ServerURL + "/trade_result", headers, 5000, post_data, result_data, result_headers);

   if(res_code == 200)
   {
      string response = CharArrayToString(result_data);
      Print("📤 Trade Result Reported: ", symbol, " | Loss: ", is_loss, " | Response: ", response);
   }
   else
   {
      Print("⚠️ Failed to report trade result. Code: ", res_code);
   }
}

//+------------------------------------------------------------------+
//| Phase 2.3: AI決済判断チェック                                      |
//+------------------------------------------------------------------+
void CheckAIExitSignals()
{
   // v7.0: Clean up tracked positions that no longer exist
   CleanupTrackedPositions();

   // 保有ポジションをループ
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      // 自分のマジックナンバーのポジションのみ対象
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      long pos_type = PositionGetInteger(POSITION_TYPE);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
      double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      double volume = PositionGetDouble(POSITION_VOLUME);
      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);

      string type_str = (pos_type == POSITION_TYPE_BUY) ? "BUY" : "SELL";

      // v7.0: Look up / create tracked entry
      int idx = FindTrackedIndex(ticket);
      if(idx < 0)
      {
         AddTrackedPosition(ticket);
         idx = FindTrackedIndex(ticket);
      }
      double max_profit_seen = 0.0;
      bool   partial_closed  = false;
      if(idx >= 0)
      {
         max_profit_seen = tracked_max_profit[idx];
         partial_closed  = tracked_partial_closed[idx];
      }

      // v7.0: Collect 30 close prices for ATR calculation
      double close_prices[];
      int copied = CopyClose(symbol, PERIOD_CURRENT, 0, 30, close_prices);
      string prices_json = "[";
      if(copied > 0)
      {
         for(int p = 0; p < copied; p++)
         {
            prices_json += DoubleToString(close_prices[p], 5);
            if(p < copied - 1) prices_json += ",";
         }
      }
      prices_json += "]";

      // v7.0: JSONリクエスト（max_profit_seen, partial_closed, prices追加）
      string json_request = StringFormat(
         "{\"account_id\": %s, \"ticket\": %d, \"symbol\": \"%s\", \"position_type\": \"%s\", \"open_price\": %.5f, \"current_price\": %.5f, \"profit\": %.2f, \"volume\": %.2f, \"open_time\": %d, \"sl\": %.5f, \"tp\": %.5f, \"max_profit_seen\": %.2f, \"partial_closed\": %s, \"prices\": %s}",
         IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)),
         ticket,
         symbol,
         type_str,
         open_price,
         current_price,
         profit,
         volume,
         (long)open_time,
         sl,
         tp,
         max_profit_seen,
         partial_closed ? "true" : "false",
         prices_json
      );

      char post_data[];
      int len = StringLen(json_request);
      StringToCharArray(json_request, post_data, 0, len);

      char result_data[];
      string result_headers;
      string headers = "Content-Type: application/json\r\n";

      int res_code = WebRequest("POST", ServerURL + "/check_exit", headers, 10000, post_data, result_data, result_headers);

      if(res_code == 200)
      {
         string response = CharArrayToString(result_data);
         string action = ParseJsonString(response, "action");
         string reason = ParseJsonString(response, "reason");

         Print("🤖 AI Exit Check [", symbol, " #", ticket, "]: ", action, " - ", reason);

         if(action == "CLOSE")
         {
            Print("📤 AI recommends closing position #", ticket);
            if(ClosePositionByTicket(ticket))
            {
               Print("✅ Position #", ticket, " closed by AI decision");
               RemoveTrackedPosition(ticket);
            }
            else
            {
               Print("❌ Failed to close position #", ticket);
            }
         }
         else if(action == "MODIFY_SL")
         {
            double new_sl = ParseJsonDouble(response, "new_sl");
            if(new_sl > 0)
            {
               // Select the position to read current TP
               if(PositionSelectByTicket(ticket))
               {
                  ModifySL(ticket, new_sl);
                  Print("✅ SL modified to ", new_sl, " for #", ticket);
               }
            }
         }
         else if(action == "PARTIAL_CLOSE")
         {
            double partial_ratio = ParseJsonDouble(response, "partial_ratio");
            if(partial_ratio > 0 && partial_ratio < 1.0)
            {
               if(PartialCloseByTicket(ticket, partial_ratio))
               {
                  Print("✅ Partial close (", partial_ratio * 100, "%) for #", ticket);
                  // Mark as partially closed
                  if(idx >= 0)
                     tracked_partial_closed[idx] = true;
               }
               else
               {
                  Print("❌ Failed partial close for #", ticket);
               }
            }
         }

         // v7.0: Update max_profit_seen
         if(idx >= 0 && profit > tracked_max_profit[idx])
            tracked_max_profit[idx] = profit;
      }
      else
      {
         Print("⚠️ AI Exit Check failed for #", ticket, ". Code: ", res_code);
      }
   }
}

//+------------------------------------------------------------------+
//| 個別ポジション決済                                                  |
//+------------------------------------------------------------------+
bool ClosePositionByTicket(ulong ticket)
{
   // ポジションを選択
   if(!PositionSelectByTicket(ticket))
   {
      Print("Error: Cannot select position #", ticket);
      return false;
   }

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action   = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol   = PositionGetString(POSITION_SYMBOL);
   request.volume   = PositionGetDouble(POSITION_VOLUME);
   request.deviation= 50;
   request.type_filling = GetFillingMode();

   long type = PositionGetInteger(POSITION_TYPE);
   if(type == POSITION_TYPE_BUY)
   {
      request.type  = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(request.symbol, SYMBOL_BID);
   }
   else
   {
      request.type  = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(request.symbol, SYMBOL_ASK);
   }

   if(OrderSend(request, result))
   {
      return true;
   }
   else
   {
      Print("Close Position Error: ", GetLastError(), " | RetCode: ", result.retcode);
      return false;
   }
}

//+------------------------------------------------------------------+
//| v7.0: 分割決済                                                     |
//+------------------------------------------------------------------+
bool PartialCloseByTicket(ulong ticket, double ratio)
{
   if(!PositionSelectByTicket(ticket))
   {
      Print("Error: Cannot select position #", ticket, " for partial close");
      return false;
   }

   double total_volume = PositionGetDouble(POSITION_VOLUME);
   double min_lot = SymbolInfoDouble(PositionGetString(POSITION_SYMBOL), SYMBOL_VOLUME_MIN);
   double step_lot = SymbolInfoDouble(PositionGetString(POSITION_SYMBOL), SYMBOL_VOLUME_STEP);

   double close_volume = total_volume * ratio;

   // Round to broker's volume step
   close_volume = min_lot + MathFloor((close_volume - min_lot) / step_lot) * step_lot;

   // If close_volume < min_lot → close entire position
   if(close_volume < min_lot)
      close_volume = total_volume;

   // If remaining < min_lot → close entire position
   double remaining = total_volume - close_volume;
   if(remaining > 0 && remaining < min_lot)
      close_volume = total_volume;

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action   = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol   = PositionGetString(POSITION_SYMBOL);
   request.volume   = close_volume;
   request.deviation= 50;
   request.type_filling = GetFillingMode();

   long type = PositionGetInteger(POSITION_TYPE);
   if(type == POSITION_TYPE_BUY)
   {
      request.type  = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(request.symbol, SYMBOL_BID);
   }
   else
   {
      request.type  = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(request.symbol, SYMBOL_ASK);
   }

   if(OrderSend(request, result))
   {
      Print("Partial Close OK: #", ticket, " | Closed: ", close_volume, " / ", total_volume);
      return true;
   }
   else
   {
      Print("Partial Close Error: ", GetLastError(), " | RetCode: ", result.retcode);
      return false;
   }
}

//+------------------------------------------------------------------+
//| v7.0: Position tracking helper functions                          |
//+------------------------------------------------------------------+
int FindTrackedIndex(ulong ticket)
{
   for(int i = 0; i < tracked_count; i++)
   {
      if(tracked_tickets[i] == ticket)
         return i;
   }
   return -1;
}

void AddTrackedPosition(ulong ticket)
{
   if(tracked_count >= MAX_TRACKED_POSITIONS)
   {
      Print("⚠️ Tracked positions full, cannot add #", ticket);
      return;
   }
   tracked_tickets[tracked_count]        = ticket;
   tracked_max_profit[tracked_count]     = 0.0;
   tracked_partial_closed[tracked_count] = false;
   tracked_count++;
}

void RemoveTrackedPosition(ulong ticket)
{
   int idx = FindTrackedIndex(ticket);
   if(idx < 0) return;

   // Shift remaining entries down
   for(int i = idx; i < tracked_count - 1; i++)
   {
      tracked_tickets[i]        = tracked_tickets[i + 1];
      tracked_max_profit[i]     = tracked_max_profit[i + 1];
      tracked_partial_closed[i] = tracked_partial_closed[i + 1];
   }
   tracked_count--;
}

void CleanupTrackedPositions()
{
   // Remove entries for positions that no longer exist
   for(int i = tracked_count - 1; i >= 0; i--)
   {
      if(!PositionSelectByTicket(tracked_tickets[i]))
      {
         // Position no longer exists, remove from tracking
         for(int j = i; j < tracked_count - 1; j++)
         {
            tracked_tickets[j]        = tracked_tickets[j + 1];
            tracked_max_profit[j]     = tracked_max_profit[j + 1];
            tracked_partial_closed[j] = tracked_partial_closed[j + 1];
         }
         tracked_count--;
      }
   }
}