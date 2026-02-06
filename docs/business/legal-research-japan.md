# Legal Research: Selling Trading Bot Software in Japan

## Overview

This document summarizes the legal requirements and business considerations for selling AI-powered trading bot software (EA) in Japan.

---

## Japan Legal Framework

### Regulatory Bodies
- **FSA (Financial Services Agency / 金融庁)**: Primary regulator
- **FIEA (Financial Instruments and Exchange Act / 金融商品取引法)**: Main law

### License Types
1. **第一種金融商品取引業** - Type I Financial Instruments Business
2. **第二種金融商品取引業** - Type II Financial Instruments Business
3. **投資運用業** - Investment Management Business
4. **投資助言・代理業** - Investment Advisory & Agency Business

---

## When License is NOT Required

Selling EA software WITHOUT registration is possible when:

| Condition | Explanation |
|-----------|-------------|
| One-time purchase (売り切り型) | User pays once, downloads, done |
| No membership required | Anyone can buy without signing up |
| No ongoing data/support | Software works standalone |
| Publicly available | Anyone can access website |

**Source:** "売り切りでの自動売買ツールの販売（インストール型）で、購入に会員登録を要しない場合は、投資助言業登録が不要になる"

---

## When License IS Required (投資助言・代理業)

Registration required when:

| Trigger | Why |
|---------|-----|
| Subscription model | Ongoing advisory relationship |
| Membership required to purchase | Not "publicly available" |
| Ongoing updates/signals provided | Continuous investment advice |
| Trading support included | Advisory service |

**Source:** "会員登録等を行わないと投資情報等を購入・利用できない場合には登録が必要となる"

---

## Registration Requirements (If Needed)

| Item | Cost/Requirement |
|------|------------------|
| Registration fee (登録免許税) | ¥150,000 |
| Security deposit (営業保証金) | ¥5,000,000 |
| Company type | 株式会社, 合同会社, or 個人 OK |
| Staff | Must have experienced personnel |
| **Total startup** | **~¥5,150,000+** |

---

## Warning: Enforcement is Increasing

> "近時、金融庁は、無登録での投資助言業ビジネスの取締りを強化する動きが見られており、無登録業者に対する緊急差し止め命令がなされた例も出てきているほか、逮捕事例も出てきている。"

**Translation:** FSA is cracking down. People have been ARRESTED for selling without proper registration.

---

## Real-World Performance vs Backtest

### The Gap
| Backtest | Real Life |
|----------|-----------|
| Perfect execution | Slippage, latency |
| No fees | Fees eat profit |
| Historical conditions | Markets change |
| 80% win rate | Often drops to 50-60% |

### Documented Results (2024-2025)
- StockHero: 76% win rate on day trading
- Grid bots: +9.6% to +21.88% gains in down markets
- AI adaptive bots: 9.2% better than single-strategy bots
- PPO optimization: 70.5% win rate, 21.3% annual return

### Hidden Costs
> "A retail trader running a bot executing 50 trades/day on a $200 balance, with each round trip costing 0.25% in fees and 0.15% in slippage, incurs daily costs exceeding $4."

---

## Why Universal Settings Don't Work

Markets are dynamic:
- Trending vs ranging conditions
- Different volatility levels
- Time-of-day variations (Asian vs London session)
- Each currency pair behaves differently

> "NO one system will work in all conditions! Market can move in different ways... strong trend, bouncy trend, flat for days/weeks, or whipsawing!"

---

## Why Goldman Sachs Settings Won't Work for Retail

| Goldman Sachs | Retail Trader |
|---------------|---------------|
| $100M+ on technology | A laptop |
| Co-located servers (microseconds) | Internet (milliseconds) |
| 9,000+ engineers | Just you |
| Private data feeds | Public data |
| Moves markets | Market moves you |

---

## Market Opportunity

| Metric | Value |
|--------|-------|
| Bot services market CAGR | 33%+ |
| Market size by 2027 | $6.7 billion |
| Retail trading using algos (2025) | 70%+ |

### Success Stories
- Platform: 4,000 free users → $650 MRR in 6 weeks
- Unibot: $120M volume, $6M fees in 2 months

---

## Business Models

### 1. One-Time Purchase (売り切り型) - SAFEST IN JAPAN
- Price: ¥30,000-100,000
- No license required
- No recurring revenue

### 2. Subscription (月額課金)
- Price: ¥3,000-20,000/month
- Requires registration in Japan
- Recurring revenue

### 3. Performance Fee (成功報酬)
- 10-20% of profits
- Definitely requires registration
- Aligned incentives

### 4. Freemium
- Free: Paper trading
- Paid: Real money trading

---

## Pricing Examples (Global Market)

| Platform | Model | Price |
|----------|-------|-------|
| 3Commas | Subscription | $19-99/month |
| Cryptohopper | Subscription | $19-99/month |
| Pionex | Free + trading fees | 0% subscription |
| TradersPost | Freemium | Free + paid tiers |

---

## Required Disclaimers (Japan)

```
【重要事項】
・本ソフトウェアは投資助言ではありません
・過去の実績は将来の利益を保証するものではありません
・投資には元本割れのリスクがあります
・自己責任でご利用ください
```

---

## Sources

### Japan Legal
- [金融庁 - 投資運用業等登録手続ガイドブック](https://www.fsa.go.jp/policy/marketentry/guidebook/index.html)
- [行政書士トーラス - EA自動売買と金融商品取引法](https://taurus-financial.com/ea-systemtrade/)
- [牛島総合法律事務所 - 投資助言業登録の判断](https://www.ushijima-law.gr.jp/topics/20250421investment_advisory/)
- [Chambers - Financial Services Regulation Japan 2025](https://practiceguides.chambers.com/practice-guides/financial-services-regulation-2025/japan/trends-and-developments)

### Business & Performance
- [3Commas - AI Trading Bot Performance](https://3commas.io/blog/ai-trading-bot-performance-analysis)
- [StockBrokers - AI Trading Bots 2026](https://www.stockbrokers.com/guides/ai-stock-trading-bots)
- [RNDpoint - Trading Bot Business Guide](https://rndpoint.com/blog/trading-bot-business-guide/)
- [AngelHack - AI Trading Bot Monetization](https://devlabs.angelhack.com/blog/ai-trading-bots/)
- [ScienceDirect - Deep Learning for Algorithmic Trading](https://www.sciencedirect.com/science/article/pii/S2590005625000177)

---

*Document created: 2026-02-03*
*Status: Research phase*
