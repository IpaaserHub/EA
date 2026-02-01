"""
複数回バックテスト実行
実データで複数回テストして安定性を確認
"""

import random
from test_historical import run_historical_backtest

def run_multiple_tests(num_runs: int = 10, deposit: int = 50000):
    """
    複数回バックテストを実行して統計を取る
    """
    results = []

    print("=" * 70)
    print(f"[複数回バックテスト] {num_runs}回実行")
    print("=" * 70)
    print(f"  入金額: {deposit:,} JPY")
    print(f"  シンボル: XAUJPY (H1)")
    print("=" * 70)

    for i in range(num_runs):
        # 毎回異なるシードで実行
        seed = 42 + i * 100
        random.seed(seed)

        print(f"\n--- Run {i+1}/{num_runs} (seed={seed}) ---")

        # バックテスト実行（出力を抑制）
        run_results = run_historical_backtest(
            symbols=["XAUJPY"],
            max_trades=500,
            deposit=deposit
        )

        if run_results:
            # 結果を集計
            total_profit = sum(r["profit"] for r in run_results)
            wins = [r for r in run_results if r["profit"] > 0]
            losses = [r for r in run_results if r["profit"] <= 0]
            total_wins = sum(r["profit"] for r in wins) if wins else 0
            total_losses = abs(sum(r["profit"] for r in losses)) if losses else 0.01
            pf = total_wins / total_losses if total_losses > 0 else float('inf')
            win_rate = len(wins) / len(run_results) * 100 if run_results else 0
            roi = (total_profit / deposit) * 100

            results.append({
                "run": i + 1,
                "seed": seed,
                "trades": len(run_results),
                "wins": len(wins),
                "win_rate": win_rate,
                "pf": pf,
                "profit": total_profit,
                "roi": roi
            })

            print(f"  取引: {len(run_results)} | 勝率: {win_rate:.1f}% | PF: {pf:.2f} | 損益: {total_profit:+,.0f} JPY | ROI: {roi:+.1f}%")

    # 統計サマリー
    if results:
        print("\n" + "=" * 70)
        print("[統計サマリー]")
        print("=" * 70)

        avg_trades = sum(r["trades"] for r in results) / len(results)
        avg_win_rate = sum(r["win_rate"] for r in results) / len(results)
        avg_pf = sum(r["pf"] for r in results) / len(results)
        avg_profit = sum(r["profit"] for r in results) / len(results)
        avg_roi = sum(r["roi"] for r in results) / len(results)

        min_profit = min(r["profit"] for r in results)
        max_profit = max(r["profit"] for r in results)
        min_roi = min(r["roi"] for r in results)
        max_roi = max(r["roi"] for r in results)

        # 勝ち回数（プラス終了）
        positive_runs = len([r for r in results if r["profit"] > 0])

        print(f"  実行回数: {num_runs}")
        print(f"  平均取引数: {avg_trades:.1f}")
        print(f"  平均勝率: {avg_win_rate:.1f}%")
        print(f"  平均PF: {avg_pf:.2f}")
        print(f"  平均損益: {avg_profit:+,.0f} JPY")
        print(f"  平均ROI: {avg_roi:+.1f}%")
        print("-" * 70)
        print(f"  最小損益: {min_profit:+,.0f} JPY ({min_roi:+.1f}%)")
        print(f"  最大損益: {max_profit:+,.0f} JPY ({max_roi:+.1f}%)")
        print(f"  プラス終了: {positive_runs}/{num_runs} ({positive_runs/num_runs*100:.0f}%)")
        print("=" * 70)

        # 詳細結果
        print("\n[各回の結果]")
        print(f"{'Run':<5} {'取引':<6} {'勝率':<8} {'PF':<8} {'損益':<14} {'ROI':<10}")
        print("-" * 60)
        for r in results:
            pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "∞"
            print(f"{r['run']:<5} {r['trades']:<6} {r['win_rate']:.1f}%{'':<3} {pf_str:<8} {r['profit']:+,.0f} JPY{'':<3} {r['roi']:+.1f}%")
        print("=" * 70)

    return results


if __name__ == "__main__":
    run_multiple_tests(num_runs=10, deposit=50000)
