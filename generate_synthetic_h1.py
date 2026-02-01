"""
合成H1データ生成スクリプト
既存のH1データの特性を基に、9月〜11月のデータを生成
"""

import os
import random
from datetime import datetime, timedelta

def analyze_existing_data(csv_path: str) -> dict:
    """既存データの特性を分析"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]  # ヘッダースキップ

    changes = []
    prices = []

    for line in lines:
        row = line.strip().split(',')
        if len(row) >= 6:
            try:
                close = float(row[4])
                change_pips = float(row[5])
                prices.append(close)
                changes.append(change_pips)
            except:
                continue

    # 統計情報
    avg_change = sum(changes) / len(changes) if changes else 0
    max_change = max(changes) if changes else 0
    min_change = min(changes) if changes else 0

    # 変動幅の標準偏差（ボラティリティ）
    avg_abs_change = sum(abs(c) for c in changes) / len(changes) if changes else 1000

    print(f"分析結果:")
    print(f"  データ数: {len(prices)}")
    print(f"  価格範囲: {min(prices):.0f} - {max(prices):.0f}")
    print(f"  平均変動: {avg_change:.0f} pips")
    print(f"  平均絶対変動: {avg_abs_change:.0f} pips")
    print(f"  最大上昇: {max_change:.0f} pips")
    print(f"  最大下落: {min_change:.0f} pips")

    return {
        "avg_abs_change": avg_abs_change,
        "max_change": max_change,
        "min_change": min_change,
        "latest_price": prices[0] if prices else 650000,  # データは降順
        "earliest_price": prices[-1] if prices else 648000,
    }


def generate_synthetic_h1_data(
    start_date: datetime,
    end_date: datetime,
    start_price: float,
    end_price: float,
    avg_volatility: float = 1500
) -> list:
    """
    合成H1データを生成

    Args:
        start_date: 開始日時
        end_date: 終了日時
        start_price: 開始価格
        end_price: 終了価格
        avg_volatility: 平均ボラティリティ（pips）

    Returns:
        list: [(date_str, open, high, low, close, change_pips, change_pct), ...]
    """
    data = []
    current_date = start_date
    current_price = start_price

    # 期間中の価格変動トレンド
    total_hours = int((end_date - start_date).total_seconds() / 3600)
    price_diff = end_price - start_price
    hourly_trend = price_diff / total_hours if total_hours > 0 else 0

    prev_close = start_price

    while current_date < end_date:
        # 週末スキップ（土日）
        if current_date.weekday() >= 5:  # 5=土曜, 6=日曜
            current_date += timedelta(hours=1)
            continue

        # 取引時間外スキップ（簡易版：22:00-23:00は休場）
        if current_date.hour == 22:
            current_date += timedelta(hours=1)
            continue

        # ランダムな変動（トレンド + ノイズ）
        noise = random.gauss(0, avg_volatility)
        change = hourly_trend + noise

        # 極端な変動を制限
        change = max(min(change, avg_volatility * 3), -avg_volatility * 3)

        # OHLCを生成
        open_price = prev_close
        close_price = open_price + change

        # High/Lowはランダムに
        intra_vol = abs(change) * random.uniform(0.5, 1.5)
        if change > 0:
            high_price = close_price + intra_vol * random.uniform(0, 0.5)
            low_price = open_price - intra_vol * random.uniform(0, 0.3)
        else:
            high_price = open_price + intra_vol * random.uniform(0, 0.3)
            low_price = close_price - intra_vol * random.uniform(0, 0.5)

        # 価格が負にならないよう
        close_price = max(close_price, 400000)
        high_price = max(high_price, close_price)
        low_price = min(low_price, close_price)
        low_price = max(low_price, 400000)

        # 変動率
        change_pips = close_price - open_price
        change_pct = (change_pips / open_price * 100) if open_price > 0 else 0

        # 日付フォーマット: MM/DD/YYYY HH:MM
        date_str = current_date.strftime("%m/%d/%Y %H:%M")

        data.append({
            "date": date_str,
            "open": round(open_price, 1),
            "high": round(high_price, 1),
            "low": round(low_price, 1),
            "close": round(close_price, 1),
            "change_pips": round(change_pips, 1),
            "change_pct": round(change_pct, 2),
        })

        prev_close = close_price
        current_date += timedelta(hours=1)

    return data


def main():
    # 既存データのパス
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    existing_h1_path = os.path.join(
        data_dir, "XAUJPY_historical_data", "9", "1-26", "1",
        "XAUJPY1時間足25", "9", "1-26", "1", "XAUJPY-1H-9.1-1.5.csv"
    )

    # 既存データを分析
    print("=== 既存H1データを分析 ===")
    stats = analyze_existing_data(existing_h1_path)

    # 合成データ生成
    print("\n=== 合成H1データを生成 ===")

    # D1データから各月の価格レベルを参考にする
    # 9月: 507,000 → 572,000
    # 10月: 568,000 → 658,000
    # 11月: 610,000 → 658,000（11/24時点で648,000）

    monthly_prices = [
        # (開始日, 終了日, 開始価格, 終了価格)
        (datetime(2025, 9, 1, 0, 0), datetime(2025, 10, 1, 0, 0), 507000, 572000),
        (datetime(2025, 10, 1, 0, 0), datetime(2025, 11, 1, 0, 0), 572000, 635000),
        (datetime(2025, 11, 1, 0, 0), datetime(2025, 11, 24, 21, 0), 635000, 648700),
    ]

    all_data = []

    for start_dt, end_dt, start_p, end_p in monthly_prices:
        print(f"  {start_dt.strftime('%Y/%m/%d')} - {end_dt.strftime('%Y/%m/%d')}: {start_p:.0f} → {end_p:.0f}")
        month_data = generate_synthetic_h1_data(
            start_date=start_dt,
            end_date=end_dt,
            start_price=start_p,
            end_price=end_p,
            avg_volatility=stats["avg_abs_change"]
        )
        all_data.extend(month_data)

    print(f"\n生成データ数: {len(all_data)} bars")

    # 既存データを読み込んで結合
    print("\n=== 既存データと結合 ===")
    with open(existing_h1_path, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    # 既存データをパース（降順のまま）
    existing_data = []
    for line in existing_lines[2:]:
        row = line.strip().split(',')
        if len(row) >= 7:
            existing_data.append({
                "date": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "change_pips": float(row[5]),
                "change_pct": float(row[6].rstrip(',')),
            })

    print(f"  既存データ: {len(existing_data)} bars")
    print(f"  合成データ: {len(all_data)} bars")

    # 合成データ（昇順）を降順に変換
    all_data.reverse()

    # 結合（既存データの後に合成データ）
    # 既存データは新しい順、合成データも新しい順に変換済み
    combined_data = existing_data + all_data

    print(f"  結合後: {len(combined_data)} bars")

    # 新しいCSVファイルに出力
    output_path = os.path.join(data_dir, "XAUJPY-H1-extended-synthetic.csv")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("XAUJPY Historical Data (Extended with Synthetic)\n")
        f.write("Date,Open,High,Low,Close,Change(Pips),Change(%)\n")

        for row in combined_data:
            f.write(f"{row['date']},{row['open']},{row['high']},{row['low']},{row['close']},{row['change_pips']},{row['change_pct']},\n")

    print(f"\n=== 出力完了 ===")
    print(f"  ファイル: {output_path}")

    # 日付範囲を確認
    first_date = combined_data[-1]["date"]
    last_date = combined_data[0]["date"]
    print(f"  日付範囲: {first_date} ～ {last_date}")


if __name__ == "__main__":
    random.seed(42)  # 再現性のため
    main()
