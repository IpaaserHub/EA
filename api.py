#!/usr/bin/env python3
"""
EA Parameter Optimizer - API Server
====================================
FastAPI server that exposes optimizer functionality for the web UI.

Run with: python api.py
Then open: http://localhost:8000
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.param_manager import ParamManager, PARAM_LIMITS, DEFAULT_PARAMS
from backtest.data_loader import load_prices, find_data_file
from backtest.engine import BacktestEngine
from optimizer.optimization_loop import OptimizationLoop, print_run_summary
from scheduler.optimizer_scheduler import OptimizerScheduler

# ==================== Configuration ====================

DATA_DIR = "data"
CONFIG_DIR = "config/params"
LOG_DIR = "logs/optimizer"

app = FastAPI(
    title="EA Parameter Optimizer",
    description="Self-improving parameter optimization for MetaTrader 5",
    version="1.0.0",
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking optimization jobs
optimization_jobs: Dict[str, Dict[str, Any]] = {}
param_manager = ParamManager(CONFIG_DIR)


# ==================== Pydantic Models ====================

class ParameterUpdate(BaseModel):
    params: Dict[str, Any]


class OptimizationRequest(BaseModel):
    symbol: str
    n_trials: int = 50
    auto_apply: bool = False
    use_ai: bool = False


class BacktestRequest(BaseModel):
    symbol: str
    params: Optional[Dict[str, Any]] = None


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Serve the dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return {"message": "EA Parameter Optimizer API", "docs": "/docs"}


@app.get("/api/status")
async def get_status():
    """Get overall system status."""
    symbols = []

    if os.path.exists(CONFIG_DIR):
        for f in os.listdir(CONFIG_DIR):
            if f.endswith('.json') and not f.startswith('optimization'):
                sym = f.replace('.json', '')
                params = param_manager.load(sym)
                data_file = find_data_file(sym, DATA_DIR)
                symbols.append({
                    "symbol": sym,
                    "has_data": data_file is not None,
                    "adx_threshold": params.get("adx_threshold"),
                    "tp_mult": params.get("tp_mult"),
                    "sl_mult": params.get("sl_mult"),
                })

    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "active_jobs": len([j for j in optimization_jobs.values() if j["status"] == "running"]),
        "param_limits": PARAM_LIMITS,
    }


@app.get("/api/symbols")
async def get_symbols():
    """Get list of configured symbols."""
    symbols = []

    if os.path.exists(CONFIG_DIR):
        for f in os.listdir(CONFIG_DIR):
            if f.endswith('.json') and not f.startswith('optimization'):
                symbols.append(f.replace('.json', ''))

    return {"symbols": sorted(symbols)}


@app.get("/api/symbols/{symbol}")
async def get_symbol_details(symbol: str):
    """Get detailed info for a symbol."""
    params = param_manager.load(symbol)
    data_file = find_data_file(symbol, DATA_DIR)

    candle_count = 0
    if data_file:
        try:
            prices = load_prices(data_file)
            candle_count = len(prices)
        except:
            pass

    history = param_manager.get_history(symbol, limit=10)

    return {
        "symbol": symbol,
        "params": params,
        "param_limits": PARAM_LIMITS,
        "has_data": data_file is not None,
        "data_file": os.path.basename(data_file) if data_file else None,
        "candle_count": candle_count,
        "history": history,
    }


@app.put("/api/symbols/{symbol}/params")
async def update_params(symbol: str, update: ParameterUpdate):
    """Update parameters for a symbol."""
    try:
        param_manager.save(symbol, update.params, reason="manual update via UI")
        return {"status": "success", "params": param_manager.load(symbol)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/symbols/{symbol}/rollback")
async def rollback_symbol(symbol: str, steps: int = 1):
    """Rollback parameters to previous version."""
    result = param_manager.rollback(symbol, steps)
    if result:
        return {"status": "success", "params": result}
    raise HTTPException(status_code=400, detail="Could not rollback - not enough history")


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest for a symbol."""
    data_file = find_data_file(request.symbol, DATA_DIR)
    if not data_file:
        raise HTTPException(status_code=404, detail=f"No data file found for {request.symbol}")

    try:
        prices = load_prices(data_file)
        params = request.params or param_manager.load(request.symbol)

        engine = BacktestEngine(prices)
        result = engine.run(params)

        # Get trade details for chart
        trades_data = []
        for trade in result.trades[-100:]:  # Last 100 trades
            trades_data.append({
                "entry_idx": trade.entry_index,
                "exit_idx": trade.exit_index,
                "direction": trade.direction.lower(),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price or trade.entry_price,
                "profit": trade.profit,
                "is_win": trade.profit > 0,
            })

        return {
            "symbol": request.symbol,
            "params": params,
            "result": {
                "total_trades": result.total_trades,
                "wins": result.wins,
                "losses": result.losses,
                "win_rate": round(result.win_rate, 2),
                "profit_factor": round(result.profit_factor, 2),
                "total_profit": round(result.total_profit, 2),
                "max_drawdown": round(result.max_drawdown, 2),
                "avg_win": round(result.avg_win, 2),
                "avg_loss": round(result.avg_loss, 2),
            },
            "trades": trades_data,
            "candle_count": len(prices),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start an optimization job."""
    job_id = f"{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    optimization_jobs[job_id] = {
        "id": job_id,
        "symbol": request.symbol,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "n_trials": request.n_trials,
        "progress": 0,
        "result": None,
    }

    background_tasks.add_task(
        run_optimization_job,
        job_id,
        request.symbol,
        request.n_trials,
        request.auto_apply,
        request.use_ai,
    )

    return {"job_id": job_id, "status": "started"}


async def run_optimization_job(
    job_id: str,
    symbol: str,
    n_trials: int,
    auto_apply: bool,
    use_ai: bool,
):
    """Background task to run optimization."""
    try:
        loop = OptimizationLoop(
            data_dir=DATA_DIR,
            config_dir=CONFIG_DIR,
            use_ai=use_ai,
        )

        run = loop.optimize_symbol(
            symbol,
            n_trials=n_trials,
            auto_apply=auto_apply,
        )

        optimization_jobs[job_id]["status"] = "completed"
        optimization_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        optimization_jobs[job_id]["result"] = {
            "old_profit_factor": round(run.old_result.profit_factor, 4),
            "new_profit_factor": round(run.new_result.profit_factor, 4),
            "improvement_pct": round(run.improvement_pct, 2),
            "applied": run.applied,
            "reason": run.reason,
            "old_params": run.old_params,
            "new_params": run.new_params,
        }

    except Exception as e:
        optimization_jobs[job_id]["status"] = "failed"
        optimization_jobs[job_id]["error"] = str(e)


@app.get("/api/jobs")
async def get_jobs():
    """Get all optimization jobs."""
    return {"jobs": list(optimization_jobs.values())}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a specific optimization job."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return optimization_jobs[job_id]


@app.get("/api/logs")
async def get_logs(limit: int = 10):
    """Get recent optimization logs."""
    logs = []

    if os.path.exists(LOG_DIR):
        log_files = sorted(
            [f for f in os.listdir(LOG_DIR) if f.endswith('.json')],
            reverse=True
        )[:limit]

        for filename in log_files:
            filepath = os.path.join(LOG_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    logs.append(json.load(f))
            except:
                pass

    return {"logs": logs}


@app.get("/api/prices/{symbol}")
async def get_prices(symbol: str, limit: int = 500):
    """Get price data for charting."""
    data_file = find_data_file(symbol, DATA_DIR)
    if not data_file:
        raise HTTPException(status_code=404, detail=f"No data file found for {symbol}")

    try:
        prices = load_prices(data_file)
        # Return last N candles
        price_data = []
        for i, p in enumerate(prices[-limit:]):
            price_data.append({
                "idx": len(prices) - limit + i,
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
            })
        return {"symbol": symbol, "prices": price_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Main ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EA Parameter Optimizer - Web UI")
    print("="*60)
    print(f"\nOpen in browser: http://localhost:8000")
    print(f"API docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
