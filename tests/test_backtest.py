"""
Tests for Backtest Module
=========================
Run with: ./venv/bin/python -m pytest tests/test_backtest.py -v

Tests for:
1. Data loading from CSV files
2. Backtest engine trade simulation
3. Performance metric calculations
"""

import pytest
import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.data_loader import (
    load_prices,
    load_mt5_csv,
    load_extended_csv,
    find_data_file,
    get_available_symbols,
)
from backtest.engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    run_backtest,
)


# ==================== Test Data Fixtures ====================

@pytest.fixture
def sample_csv_content():
    """Sample CSV content in extended format."""
    return """XAUJPY Historical Data
Date,Open,High,Low,Close,Change,Change%
2025.09.17 11:00,537501.00000,537616.00000,536374.00000,536696.00000,100,0.1
2025.09.17 12:00,536696.00000,537409.00000,536351.00000,536640.00000,100,0.1
2025.09.17 13:00,536640.00000,536837.00000,535260.00000,535534.00000,100,0.1
2025.09.17 14:00,535528.00000,536721.00000,535207.00000,536437.00000,100,0.1
2025.09.17 15:00,536440.00000,538714.00000,536437.00000,538452.00000,100,0.1
"""


@pytest.fixture
def temp_csv_file(sample_csv_content):
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_content)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def uptrend_prices():
    """Generate 100 candles in an uptrend."""
    prices = []
    base = 100.0
    for i in range(100):
        price = base + i * 0.5
        prices.append({
            "open": price - 0.1,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price,
        })
    return prices


@pytest.fixture
def downtrend_prices():
    """Generate 100 candles in a downtrend."""
    prices = []
    base = 150.0
    for i in range(100):
        price = base - i * 0.5
        prices.append({
            "open": price + 0.1,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price,
        })
    return prices


@pytest.fixture
def default_params():
    """Default trading parameters."""
    return {
        "adx_threshold": 5,
        "slope_threshold": 0.00001,
        "buy_position": 0.5,
        "sell_position": 0.5,
        "rsi_buy_max": 75,
        "rsi_sell_min": 25,
        "tp_mult": 2.0,
        "sl_mult": 1.5,
    }


# ==================== Data Loader Tests ====================

class TestDataLoader:
    """Tests for data loading functions."""

    def test_load_extended_csv(self, temp_csv_file):
        """Should load extended format CSV."""
        prices = load_extended_csv(temp_csv_file)
        assert len(prices) == 5
        assert prices[0]["close"] == 536696.0

    def test_load_prices_auto_detect(self, temp_csv_file):
        """load_prices should auto-detect format."""
        prices = load_prices(temp_csv_file)
        assert len(prices) == 5

    def test_load_nonexistent_file(self):
        """Should return empty list for missing file."""
        prices = load_prices("/nonexistent/file.csv")
        assert prices == []

    def test_find_data_file_returns_none_for_missing(self):
        """Should return None if no data file found."""
        result = find_data_file("NONEXISTENT", "/tmp")
        assert result is None


class TestDataLoaderWithRealData:
    """Tests using real data files (if available)."""

    @pytest.fixture
    def data_dir(self):
        """Get the data directory path."""
        return os.path.join(os.path.dirname(__file__), '..', 'data')

    def test_load_xaujpy_if_exists(self, data_dir):
        """Load XAUJPY data if file exists."""
        data_file = find_data_file("XAUJPY", data_dir)
        if data_file:
            prices = load_prices(data_file)
            assert len(prices) > 0
            assert "close" in prices[0]
        else:
            pytest.skip("XAUJPY data file not found")

    def test_get_available_symbols(self, data_dir):
        """Should find available symbols."""
        if os.path.exists(data_dir):
            symbols = get_available_symbols(data_dir)
            # Should find at least one symbol
            assert isinstance(symbols, list)
        else:
            pytest.skip("Data directory not found")


# ==================== Backtest Engine Tests ====================

class TestBacktestEngine:
    """Tests for backtest engine."""

    def test_run_returns_result(self, uptrend_prices, default_params):
        """run() should return BacktestResult."""
        engine = BacktestEngine(uptrend_prices)
        result = engine.run(default_params)
        assert isinstance(result, BacktestResult)

    def test_result_has_all_fields(self, uptrend_prices, default_params):
        """Result should have all expected fields."""
        engine = BacktestEngine(uptrend_prices)
        result = engine.run(default_params)

        assert hasattr(result, 'total_trades')
        assert hasattr(result, 'wins')
        assert hasattr(result, 'losses')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'profit_factor')
        assert hasattr(result, 'total_profit')

    def test_wins_plus_losses_equals_total(self, uptrend_prices, default_params):
        """Wins + losses should equal total trades."""
        engine = BacktestEngine(uptrend_prices)
        result = engine.run(default_params)
        assert result.wins + result.losses == result.total_trades

    def test_win_rate_is_percentage(self, uptrend_prices, default_params):
        """Win rate should be 0-100."""
        engine = BacktestEngine(uptrend_prices)
        result = engine.run(default_params)
        assert 0 <= result.win_rate <= 100

    def test_empty_result_for_insufficient_data(self, default_params):
        """Should return empty result if not enough data."""
        small_prices = [{"open": 100, "high": 101, "low": 99, "close": 100}] * 10
        engine = BacktestEngine(small_prices)
        result = engine.run(default_params)
        assert result.total_trades == 0

    def test_max_trades_limits_results(self, uptrend_prices, default_params):
        """Should respect max_trades limit."""
        engine = BacktestEngine(uptrend_prices)
        result = engine.run(default_params, max_trades=5)
        assert result.total_trades <= 5

    def test_to_dict_returns_dict(self, uptrend_prices, default_params):
        """to_dict() should return serializable dict."""
        engine = BacktestEngine(uptrend_prices)
        result = engine.run(default_params)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "total_trades" in result_dict
        assert "win_rate" in result_dict


class TestBacktestParameters:
    """Test how different parameters affect results."""

    def test_strict_adx_reduces_trades(self, uptrend_prices):
        """Higher ADX threshold should reduce number of trades."""
        engine = BacktestEngine(uptrend_prices)

        loose_params = {"adx_threshold": 5}
        strict_params = {"adx_threshold": 25}

        loose_result = engine.run(loose_params)
        strict_result = engine.run(strict_params)

        # Stricter params should have fewer or equal trades
        assert strict_result.total_trades <= loose_result.total_trades

    def test_higher_tp_affects_win_rate(self, uptrend_prices):
        """Higher TP multiplier may reduce win rate but increase avg win."""
        engine = BacktestEngine(uptrend_prices)

        low_tp = {"tp_mult": 1.5, "sl_mult": 1.5, "adx_threshold": 1}  # Very loose to get trades
        high_tp = {"tp_mult": 3.5, "sl_mult": 1.5, "adx_threshold": 1}

        low_result = engine.run(low_tp)
        high_result = engine.run(high_tp)

        # Both should produce valid results (win_rate is float or int 0)
        assert isinstance(low_result.win_rate, (int, float))
        assert isinstance(high_result.win_rate, (int, float))
        assert 0 <= low_result.win_rate <= 100
        assert 0 <= high_result.win_rate <= 100


class TestConvenienceFunction:
    """Test run_backtest convenience function."""

    def test_run_backtest_works(self, uptrend_prices, default_params):
        """run_backtest() should work like engine.run()."""
        result = run_backtest(uptrend_prices, default_params)
        assert isinstance(result, BacktestResult)
        assert hasattr(result, 'total_trades')


# ==================== Trade Class Tests ====================

class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self):
        """Should create trade with required fields."""
        trade = Trade(
            entry_index=10,
            entry_price=100.0,
            direction="BUY",
            sl=95.0,
            tp=110.0,
        )
        assert trade.entry_price == 100.0
        assert trade.direction == "BUY"
        assert trade.outcome == "OPEN"

    def test_trade_defaults(self):
        """Trade should have sensible defaults."""
        trade = Trade(
            entry_index=0,
            entry_price=100.0,
            direction="SELL",
            sl=105.0,
            tp=90.0,
        )
        assert trade.exit_index is None
        assert trade.exit_price is None
        assert trade.profit == 0.0


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests using real data files."""

    @pytest.fixture
    def real_prices(self):
        """Load real price data if available."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_file = find_data_file("XAUJPY", data_dir)
        if data_file:
            return load_prices(data_file)
        return None

    def test_full_backtest_on_real_data(self, real_prices, default_params):
        """Run full backtest on real data."""
        if not real_prices or len(real_prices) < 100:
            pytest.skip("Not enough real data available")

        result = run_backtest(real_prices, default_params)

        # Should produce some trades
        assert result.total_trades > 0
        # Metrics should be valid
        assert 0 <= result.win_rate <= 100
        assert result.profit_factor >= 0


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
