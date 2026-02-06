"""
Tests for News/Event Filter
============================
Run with: .venv/bin/python -m pytest tests/test_news_filter.py -v

Tests for:
1. Calendar loading
2. Event date parsing (fixed dates, first-friday schedule)
3. should_trade() pause windows
4. Currency relevance filtering
5. get_upcoming_events()
"""

import pytest
import os
import sys
import json
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.news_filter import NewsFilter


# ==================== Test Fixtures ====================

@pytest.fixture
def calendar_file(tmp_path):
    """Create a temporary calendar JSON for testing."""
    calendar = {
        "events": [
            {
                "name": "NFP",
                "currency": "USD",
                "impact": "HIGH",
                "schedule": "first_friday_monthly",
                "time_utc": "13:30",
                "pause_before_min": 60,
                "pause_after_min": 120,
            },
            {
                "name": "FOMC",
                "currency": "USD",
                "impact": "HIGH",
                "dates_2026": ["2026-03-18", "2026-06-17"],
                "time_utc": "19:00",
                "pause_before_min": 60,
                "pause_after_min": 120,
            },
            {
                "name": "BOJ Decision",
                "currency": "JPY",
                "impact": "HIGH",
                "dates_2026": ["2026-01-24", "2026-03-14"],
                "time_utc": "03:00",
                "pause_before_min": 60,
                "pause_after_min": 60,
            },
            {
                "name": "ECB Decision",
                "currency": "EUR",
                "impact": "HIGH",
                "dates_2026": ["2026-04-17"],
                "time_utc": "13:15",
                "pause_before_min": 60,
                "pause_after_min": 120,
            },
        ],
        "currency_map": {
            "USDJPY": ["USD", "JPY"],
            "XAUJPY": ["XAU", "JPY", "USD"],
            "BTCJPY": ["BTC", "JPY", "USD"],
            "EURUSD": ["EUR", "USD"],
        },
    }
    path = tmp_path / "test_calendar.json"
    with open(path, "w") as f:
        json.dump(calendar, f)
    return str(path)


@pytest.fixture
def news_filter(calendar_file):
    """Create a NewsFilter with test calendar."""
    return NewsFilter(calendar_file)


# ==================== Calendar Loading Tests ====================

class TestCalendarLoading:
    """Tests for loading the economic calendar."""

    def test_loads_events(self, news_filter):
        """Should load events from JSON."""
        assert len(news_filter.events) == 4

    def test_loads_currency_map(self, news_filter):
        """Should load currency map."""
        assert "USDJPY" in news_filter.currency_map
        assert "USD" in news_filter.currency_map["USDJPY"]
        assert "JPY" in news_filter.currency_map["USDJPY"]

    def test_missing_file_no_error(self, tmp_path):
        """Should handle missing calendar file gracefully."""
        nf = NewsFilter(str(tmp_path / "nonexistent.json"))
        assert nf.events == []
        can_trade, reason = nf.should_trade("USDJPY")
        assert can_trade is True

    def test_empty_calendar(self, tmp_path):
        """Should handle empty calendar."""
        path = tmp_path / "empty.json"
        with open(path, "w") as f:
            json.dump({"events": [], "currency_map": {}}, f)
        nf = NewsFilter(str(path))
        can_trade, reason = nf.should_trade("USDJPY")
        assert can_trade is True


# ==================== Currency Relevance Tests ====================

class TestCurrencyRelevance:
    """Tests for currency-to-symbol mapping."""

    def test_usdjpy_affected_by_usd(self, news_filter):
        """USDJPY should be affected by USD events."""
        currencies = news_filter._get_relevant_currencies("USDJPY")
        assert "USD" in currencies

    def test_usdjpy_affected_by_jpy(self, news_filter):
        """USDJPY should be affected by JPY events."""
        currencies = news_filter._get_relevant_currencies("USDJPY")
        assert "JPY" in currencies

    def test_xaujpy_affected_by_usd(self, news_filter):
        """XAUJPY should be affected by USD events (gold priced in USD)."""
        currencies = news_filter._get_relevant_currencies("XAUJPY")
        assert "USD" in currencies

    def test_eurusd_not_affected_by_jpy(self, news_filter):
        """EURUSD should not be affected by JPY events."""
        currencies = news_filter._get_relevant_currencies("EURUSD")
        assert "JPY" not in currencies

    def test_unknown_symbol_fallback(self, news_filter):
        """Unknown symbol should use fallback parsing."""
        currencies = news_filter._get_relevant_currencies("GBPUSD")
        assert "GBP" in currencies
        assert "USD" in currencies


# ==================== Should Trade Tests ====================

class TestShouldTrade:
    """Tests for the should_trade() method."""

    def test_safe_time_allows_trading(self, news_filter):
        """Should allow trading when no events are near."""
        # Random safe time (no events)
        safe_time = datetime(2026, 2, 10, 10, 0)  # Tuesday 10:00 UTC
        can_trade, reason = news_filter.should_trade("USDJPY", safe_time)
        assert can_trade is True
        assert reason == ""

    def test_blocks_before_fomc(self, news_filter):
        """Should block trading 60 min before FOMC."""
        # FOMC is 2026-03-18 19:00 UTC, pause_before=60
        before_fomc = datetime(2026, 3, 18, 18, 30)  # 30 min before
        can_trade, reason = news_filter.should_trade("USDJPY", before_fomc)
        assert can_trade is False
        assert "FOMC" in reason

    def test_blocks_after_fomc(self, news_filter):
        """Should block trading 120 min after FOMC."""
        # FOMC is 2026-03-18 19:00 UTC, pause_after=120
        after_fomc = datetime(2026, 3, 18, 20, 0)  # 60 min after
        can_trade, reason = news_filter.should_trade("USDJPY", after_fomc)
        assert can_trade is False
        assert "FOMC" in reason

    def test_allows_trading_well_after_fomc(self, news_filter):
        """Should allow trading long after FOMC."""
        long_after = datetime(2026, 3, 18, 22, 0)  # 3 hours after
        can_trade, reason = news_filter.should_trade("USDJPY", long_after)
        assert can_trade is True

    def test_blocks_before_boj_for_jpy_pair(self, news_filter):
        """Should block USDJPY before BOJ decision."""
        # BOJ is 2026-01-24 03:00 UTC, pause_before=60
        before_boj = datetime(2026, 1, 24, 2, 30)  # 30 min before
        can_trade, reason = news_filter.should_trade("USDJPY", before_boj)
        assert can_trade is False
        assert "BOJ" in reason

    def test_ecb_does_not_block_usdjpy(self, news_filter):
        """ECB event should not block USDJPY (no EUR component)."""
        # ECB is 2026-04-17 13:15 UTC
        during_ecb = datetime(2026, 4, 17, 13, 15)
        can_trade, reason = news_filter.should_trade("USDJPY", during_ecb)
        assert can_trade is True

    def test_ecb_blocks_eurusd(self, news_filter):
        """ECB event should block EURUSD."""
        during_ecb = datetime(2026, 4, 17, 13, 15)
        can_trade, reason = news_filter.should_trade("EURUSD", during_ecb)
        assert can_trade is False
        assert "ECB" in reason

    def test_nfp_first_friday(self, news_filter):
        """NFP should trigger on first Friday of month."""
        # First Friday of March 2026 is March 6
        before_nfp = datetime(2026, 3, 6, 13, 0)  # 30 min before 13:30
        can_trade, reason = news_filter.should_trade("USDJPY", before_nfp)
        assert can_trade is False
        assert "NFP" in reason

    def test_reason_shows_minutes_before(self, news_filter):
        """Reason should say 'in Xmin' before event."""
        before_fomc = datetime(2026, 3, 18, 18, 30)
        can_trade, reason = news_filter.should_trade("USDJPY", before_fomc)
        assert "in 30min" in reason

    def test_reason_shows_minutes_after(self, news_filter):
        """Reason should say 'was Xmin ago' after event."""
        after_fomc = datetime(2026, 3, 18, 19, 45)
        can_trade, reason = news_filter.should_trade("USDJPY", after_fomc)
        assert "45min ago" in reason


# ==================== Upcoming Events Tests ====================

class TestUpcomingEvents:
    """Tests for get_upcoming_events()."""

    def test_returns_upcoming(self, news_filter):
        """Should return events within the lookahead window."""
        # Day before FOMC
        check_time = datetime(2026, 3, 17, 19, 0)
        events = news_filter.get_upcoming_events("USDJPY", check_time, hours_ahead=24)
        names = [e["name"] for e in events]
        assert "FOMC" in names

    def test_sorted_by_time(self, news_filter):
        """Events should be sorted by minutes_until."""
        check_time = datetime(2026, 1, 23, 12, 0)
        events = news_filter.get_upcoming_events("USDJPY", check_time, hours_ahead=48)
        if len(events) > 1:
            for i in range(len(events) - 1):
                assert events[i]["minutes_until"] <= events[i + 1]["minutes_until"]

    def test_does_not_return_past_events(self, news_filter):
        """Should not return events that already happened."""
        after_fomc = datetime(2026, 3, 18, 20, 0)
        events = news_filter.get_upcoming_events("USDJPY", after_fomc, hours_ahead=1)
        fomc_events = [e for e in events if e["name"] == "FOMC"]
        assert len(fomc_events) == 0

    def test_returns_empty_for_no_events(self, news_filter):
        """Should return empty list when no events upcoming."""
        safe_time = datetime(2026, 2, 10, 10, 0)
        events = news_filter.get_upcoming_events("USDJPY", safe_time, hours_ahead=1)
        assert events == []


# ==================== Integration with Real Calendar ====================

class TestRealCalendar:
    """Tests using the actual economic_calendar.json if it exists."""

    def test_loads_real_calendar(self):
        """Should load real calendar without errors."""
        real_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'economic_calendar.json'
        )
        if not os.path.exists(real_path):
            pytest.skip("Real calendar not found")

        nf = NewsFilter(real_path)
        assert len(nf.events) > 0

    def test_real_calendar_all_events_have_currency(self):
        """All events in real calendar should have a currency field."""
        real_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'economic_calendar.json'
        )
        if not os.path.exists(real_path):
            pytest.skip("Real calendar not found")

        nf = NewsFilter(real_path)
        for event in nf.events:
            assert "currency" in event, f"Event {event.get('name')} missing currency"
            assert "time_utc" in event, f"Event {event.get('name')} missing time_utc"


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
