//+------------------------------------------------------------------+
//|                                              ExportHistory.mq5   |
//|                                    XAUJPY履歴データエクスポート    |
//+------------------------------------------------------------------+
#property script_show_inputs

input string   Symbol_Name = "XAUJPY";      // シンボル名
input ENUM_TIMEFRAMES TimeFrame = PERIOD_H1; // 時間足
input int      Bars_Count = 5000;            // 取得バー数

void OnStart()
{
   string filename = Symbol_Name + "_H1_export.csv";

   int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("ファイルを開けません: ", filename);
      return;
   }

   // ヘッダー（なし - データのみ）

   MqlRates rates[];
   int copied = CopyRates(Symbol_Name, TimeFrame, 0, Bars_Count, rates);

   if(copied <= 0)
   {
      Print("データ取得失敗: ", GetLastError());
      FileClose(handle);
      return;
   }

   Print("取得バー数: ", copied);

   // 古い順に出力
   for(int i = 0; i < copied; i++)
   {
      string line = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES) + "," +
                    DoubleToString(rates[i].open, 5) + "," +
                    DoubleToString(rates[i].high, 5) + "," +
                    DoubleToString(rates[i].low, 5) + "," +
                    DoubleToString(rates[i].close, 5) + "," +
                    IntegerToString(rates[i].tick_volume) + ",0";
      FileWrite(handle, line);
   }

   FileClose(handle);

   string path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + filename;
   Print("エクスポート完了: ", path);
   Print("バー数: ", copied);

   Alert("エクスポート完了!\n", path);
}
