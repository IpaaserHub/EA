"""
News/Event Filter
=================
Pauses trading before/after high-impact economic events to prevent
trades from being blown out by news spikes.

Uses a static economic calendar (JSON) — no API calls, no LLM needed.
Rule-based is better here: events are pre-categorized, decision is
binary (trade/don't trade), and this approach has zero cost and latency.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class NewsFilter:
    """
    Filters trading around high-impact economic events.

    Usage:
        filter = NewsFilter("config/economic_calendar.json")
        can_trade, reason = filter.should_trade("USDJPY", datetime.utcnow())
        if not can_trade:
            print(f"Trading paused: {reason}")
    """

    def __init__(self, calendar_path: str = "config/economic_calendar.json"):
        """
        Initialize with path to economic calendar JSON.

        Args:
            calendar_path: Path to the JSON calendar file
        """
        self.calendar_path = calendar_path
        self.events: List[Dict] = []
        self.currency_map: Dict[str, List[str]] = {}
        self._load_calendar()

    def _load_calendar(self):
        """Load economic calendar from JSON file."""
        if not os.path.exists(self.calendar_path):
            return

        with open(self.calendar_path, "r") as f:
            data = json.load(f)

        self.events = data.get("events", [])
        self.currency_map = data.get("currency_map", {})

    def _get_relevant_currencies(self, symbol: str) -> List[str]:
        """Get currencies relevant to a trading symbol."""
        if symbol in self.currency_map:
            return self.currency_map[symbol]
        # Fallback: extract from symbol name
        # Common patterns: USDJPY, XAUJPY, BTCJPY
        currencies = []
        if "USD" in symbol:
            currencies.append("USD")
        if "JPY" in symbol:
            currencies.append("JPY")
        if "EUR" in symbol:
            currencies.append("EUR")
        if "GBP" in symbol:
            currencies.append("GBP")
        if "XAU" in symbol:
            currencies.extend(["XAU", "USD"])
        if "BTC" in symbol:
            currencies.extend(["BTC", "USD"])
        return list(set(currencies))

    def _get_event_dates(self, event: Dict, year: int) -> List[str]:
        """Get all dates for an event in a given year."""
        dates_key = f"dates_{year}"
        if dates_key in event:
            return event[dates_key]

        if event.get("schedule") == "first_friday_monthly":
            return self._first_fridays(year)

        return []

    def _first_fridays(self, year: int) -> List[str]:
        """Get first Friday of each month for a given year."""
        fridays = []
        for month in range(1, 13):
            d = datetime(year, month, 1)
            # Find first Friday (weekday 4)
            while d.weekday() != 4:
                d += timedelta(days=1)
            fridays.append(d.strftime("%Y-%m-%d"))
        return fridays

    def should_trade(
        self, symbol: str, current_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Check if trading is safe right now for the given symbol.

        Args:
            symbol: Trading symbol (e.g., "USDJPY")
            current_time: Current UTC time (defaults to now)

        Returns:
            Tuple of (can_trade: bool, reason: str)
            - (True, "") if trading is safe
            - (False, reason) if trading should be paused
        """
        if current_time is None:
            current_time = datetime.utcnow()

        relevant_currencies = self._get_relevant_currencies(symbol)
        if not relevant_currencies:
            return True, ""

        year = current_time.year
        current_date_str = current_time.strftime("%Y-%m-%d")

        for event in self.events:
            # Skip events for unrelated currencies
            if event.get("currency") not in relevant_currencies:
                continue

            event_dates = self._get_event_dates(event, year)

            for date_str in event_dates:
                # Parse event datetime
                time_str = event.get("time_utc", "00:00")
                try:
                    event_dt = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                    )
                except ValueError:
                    continue

                pause_before = timedelta(
                    minutes=event.get("pause_before_min", 60)
                )
                pause_after = timedelta(
                    minutes=event.get("pause_after_min", 60)
                )

                window_start = event_dt - pause_before
                window_end = event_dt + pause_after

                if window_start <= current_time <= window_end:
                    event_name = event.get("name", "Unknown")
                    if current_time < event_dt:
                        mins_until = int(
                            (event_dt - current_time).total_seconds() / 60
                        )
                        reason = (
                            f"{event_name} in {mins_until}min — trading paused"
                        )
                    else:
                        mins_since = int(
                            (current_time - event_dt).total_seconds() / 60
                        )
                        reason = (
                            f"{event_name} was {mins_since}min ago — "
                            f"waiting for volatility to settle"
                        )
                    return False, reason

        return True, ""

    def get_upcoming_events(
        self,
        symbol: str,
        current_time: Optional[datetime] = None,
        hours_ahead: int = 24,
    ) -> List[Dict]:
        """
        Get upcoming events that could affect a symbol.

        Args:
            symbol: Trading symbol
            current_time: Current UTC time
            hours_ahead: How far ahead to look

        Returns:
            List of upcoming event dicts with datetime info
        """
        if current_time is None:
            current_time = datetime.utcnow()

        relevant_currencies = self._get_relevant_currencies(symbol)
        upcoming = []
        year = current_time.year
        lookahead = current_time + timedelta(hours=hours_ahead)

        for event in self.events:
            if event.get("currency") not in relevant_currencies:
                continue

            event_dates = self._get_event_dates(event, year)

            for date_str in event_dates:
                time_str = event.get("time_utc", "00:00")
                try:
                    event_dt = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                    )
                except ValueError:
                    continue

                if current_time <= event_dt <= lookahead:
                    upcoming.append({
                        "name": event.get("name"),
                        "currency": event.get("currency"),
                        "impact": event.get("impact"),
                        "datetime": event_dt.isoformat(),
                        "minutes_until": int(
                            (event_dt - current_time).total_seconds() / 60
                        ),
                        "pause_before_min": event.get("pause_before_min", 60),
                        "pause_after_min": event.get("pause_after_min", 60),
                    })

        upcoming.sort(key=lambda x: x["minutes_until"])
        return upcoming
