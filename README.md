# AI EA - 自動売買エキスパートアドバイザー

> **バージョン**: v10.7 (Aggressive Mode)
> **最終更新**: 2026年2月

AI搭載のFX・仮想通貨自動売買システム。MetaTrader 5と連携し、テクニカル分析とAI判断を組み合わせた高精度トレードを実現します。

---

## 概要

本システムは、以下の問題を解決するために設計されました：

- **手動トレードの課題**: 常時監視が必要、感情的な判断、タイミングのずれ
- **解決策**: ルールベースのテクニカル分析 + AI（OpenAI/Gemini）による意思決定の自動化

### 対応銘柄

| レベル | 銘柄 | 時間足 | 月間目標 | 特徴 |
|--------|------|--------|----------|------|
| Lv1 | USDJPY | H1 | ¥5万 | 低ボラティリティ、安定運用 |
| Lv2 | GBPJPY | H1/M30 | ¥5-20万 | 中ボラティリティ、トレンド狙い |
| Lv3 | XAUUSD | H1 | ¥20-50万 | 高ボラティリティ、ATR向き |
| Lv4 | BTCJPY | M15-H1 | ¥50万+ | 24時間稼働、トレンドフォロー |

---

## 技術スタック

| コンポーネント | 技術 |
|----------------|------|
| バックエンド | Python 3.x + FastAPI + Uvicorn |
| トレード実行 | MQL5 (MetaTrader 5) |
| AI連携 | OpenAI GPT-4o / Google Gemini |
| データベース | SQLite |
| フロントエンド | HTML/CSS/JavaScript (ダッシュボード) |

---

## プロジェクト構成

```
/EA/
├── main_genai_custom.py      # メインサーバー (v10.7) ⭐
├── main_advanced.py          # ベースラインサーバー (v2.1)
├── main_genai_ui.py          # UIダッシュボード版
├── main_speedy.py            # テスト用軽量版
│
├── mql5/                     # MetaTrader 5 EA
│   ├── MT5 EA - Ultimate UI (Final Fix).mq5
│   ├── MT5 EA - Ultimate UI (Final Fix).ex5
│   └── ExportHistory.mq5
│
├── data/                     # 価格履歴データ (CSV)
│   ├── BTCJPYH1.csv
│   ├── USDJPYH1.csv
│   └── ...
│
├── test_*.py                 # バックテスト・分析ツール
│   ├── test_backtest.py      # マルチペア並列バックテスト
│   ├── test_historical.py    # 長期履歴分析
│   ├── test_monthly.py       # 月次P/L追跡
│   └── ...
│
├── リードミー/               # ドキュメント
│   ├── EAロジック仕様書（アドバンス版）.md
│   ├── 目標金額別_推奨設定ガイド.md
│   ├── 実装ロードマップ_v2.md
│   └── ...
│
├── trading_log.db            # 取引ログDB
└── server.log                # サーバーログ
```

---

## セットアップ手順

### 1. 必要環境

- Python 3.9以上
- MetaTrader 5（デモまたはライブ口座）
- OpenAI APIキー または Google Gemini APIキー（オプション）

### 2. 依存パッケージのインストール

```bash
cd EA
pip install fastapi uvicorn pydantic
pip install openai google-generativeai  # AI機能を使う場合
```

### 3. 環境変数の設定

```bash
# Linux/Mac
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"

# Windows (コマンドプロンプト)
set OPENAI_API_KEY=your-openai-api-key
set GOOGLE_API_KEY=your-google-api-key
```

### 4. 口座IDの設定

`main_genai_custom.py` の24行目を編集：

```python
ALLOWED_ACCOUNTS = [あなたの口座ID]  # 例: [75449373]
```

### 5. サーバー起動

```bash
python main_genai_custom.py
```

`Uvicorn running on http://0.0.0.0:8000` と表示されれば成功。

### 6. MetaTrader 5の設定

1. `mql5/MT5 EA - Ultimate UI (Final Fix).mq5` をMT5の`Experts`フォルダにコピー
2. MT5でコンパイル
3. 運用したいチャート（例：USDJPY H1）を開く
4. EAをチャートにドラッグ＆ドロップ
5. 「自動売買」ボタンをON

---

## APIエンドポイント

| エンドポイント | メソッド | 説明 |
|----------------|----------|------|
| `/signal` | POST | 売買シグナル取得（SL/TP含む） |
| `/history` | POST | 価格履歴の同期 |
| `/check_entry` | POST | エントリー条件の確認 |
| `/check_exit` | POST | 決済判断（AI） |
| `/dashboard` | GET | 取引ダッシュボード |
| `/` | GET | ヘルスチェック |

---

## トレードロジック

### 環境認識（AIの「目」）

1. **線形回帰（Slope）**: トレンド方向を検出
2. **レンジ内位置（Position）**: 現在価格の相対位置（0-100%）
3. **ATR**: ボラティリティ測定（動的SL/TP計算用）
4. **RSI**: 買われすぎ/売られすぎ判定
5. **ADX**: トレンド強度確認

### エントリー条件

**買いシグナル (BUY)**:
- Slope > 閾値（上昇トレンド）
- Position < 0.5（安値圏）
- RSI < 75（買われすぎでない）

**売りシグナル (SELL)**:
- Slope < -閾値（下降トレンド）
- Position > 0.5（高値圏）
- RSI > 25（売られすぎでない）

### リスク管理

- **動的SL**: 直近安値/高値 ± ATR × 係数
- **動的TP**: 直近高値/安値 ± ATR × 係数
- **R:Rフィルター**: リスクリワード比 >= 1.0 のみエントリー
- **最大ポジション数**: 銘柄別に制限
- **クールダウン**: 3連敗で10-30分休止

---

## 設定パラメータ

### トレードモード

```python
TRADE_MODE = "AGGRESSIVE"  # または "STABLE"
```

| モード | 特徴 | 推奨時間足 |
|--------|------|------------|
| STABLE | 高勝率・低頻度 | M15/H1 |
| AGGRESSIVE | 高頻度・収益重視 | M5 |

### 銘柄別設定

```python
SYMBOL_CONFIG = {
    "BTCJPY": {
        "history_size": 120,      # 分析足数
        "max_positions": 1,       # 最大ポジション数
        "cooldown_minutes": 30,   # クールダウン時間
        "atr_multiplier": 0.7     # ATR係数
    },
    # ...
}
```

---

## バックテスト

```bash
# マルチペア並列バックテスト
python test_backtest.py

# 履歴分析
python test_historical.py

# パラメータ最適化
python test_sltp_comparison.py
```

---

## トラブルシューティング

| 症状 | 原因 | 対処法 |
|------|------|--------|
| Connection Error | サーバー未起動 | `python main_genai_custom.py` を実行 |
| 30秒経っても無反応 | DLL許可設定漏れ | MT5のEA設定でDLLを許可 |
| 足確定後2分以上無反応 | MT5フリーズ | MT5を再起動 |
| AI応答エラー | APIキー未設定 | 環境変数を確認 |

---

## 開発ロードマップ

### 実装済み (Phase 1)
- [x] 基本通信・取引機能
- [x] 線形回帰チャネル + ATR エントリー
- [x] 銘柄別設定（SYMBOL_CONFIG）
- [x] クールダウン機能
- [x] RSI/ADXフィルター
- [x] AIによるSL/TP決定
- [x] 決済判断のAI化

### 開発中 (Phase 2)
- [ ] Discord/LINE通知
- [ ] ダッシュボード強化
- [ ] マルチタイムフレーム分析

### 将来計画 (Phase 3-4)
- [ ] ファンダメンタルズ統合
- [ ] ニュースセンチメント分析
- [ ] 機械学習モデル置換

---

## ライセンス

プライベートプロジェクト

---

## 連絡先

開発者: IpaaserHub
