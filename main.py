"""
Run:  uvicorn dashboard_api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from dashboard_service import DashboardService
from telemetry_store import TelemetryFetchError, RESOLVER_VERSION
from investigation_persistence import PersistenceError
from planner_agent import PlannerAgent

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Cloud Optimization API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("DASHBOARD_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)

svc = DashboardService()
_running_tasks: dict[str, asyncio.Task] = {}


def _http(e: Exception) -> HTTPException:
    if isinstance(e, (TelemetryFetchError, PersistenceError)):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")



@app.get("/api/subscriptions")
def subscriptions():
    try:
        return svc.list_subscriptions()
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services")
async def services(sub: str):
    try:
        return await svc.list_services(sub)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/workspaces")
async def workspaces(sub: str, service: str):
    try:
        return await svc.list_workspaces(sub, service)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/item-types")
async def item_types(sub: str, service: str, rg: str, ws: str):
    try:
        return await svc.list_item_types(sub, service, rg, ws)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items")
async def items(sub: str, service: str, rg: str, ws: str,
                item_type: Optional[str] = Query(default=None)):
    try:
        return await svc.list_items(sub, service, rg, ws, item_type)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items/{item_type}/{item_name}")
async def item_detail(sub: str, service: str, rg: str, ws: str,
                      item_type: str, item_name: str):
    try:
        return await svc.item_detail(sub, service, rg, ws, item_type, item_name)
    except Exception as e:
        raise _http(e)



async def _run_investigation_task(payload: dict, investigation_id: str):

    try:
        planner = PlannerAgent()
        await planner.run_investigation(payload, investigation_id=investigation_id)
    except Exception as e:
        
        print(f"[analyze:{investigation_id}] failed: {type(e).__name__}: {e}")
    finally:
        _running_tasks.pop(investigation_id, None)


@app.post("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items/{item_type}/{item_name}/analyze")
async def analyze_workload(sub: str, service: str, rg: str, ws: str,
                           item_type: str, item_name: str):
    try:
        payload = await svc.build_trigger_payload(sub, service, rg, ws, item_type, item_name)
    except Exception as e:
        raise _http(e)

    investigation_id = PlannerAgent.new_investigation_id()
    task = asyncio.create_task(_run_investigation_task(payload, investigation_id))
    _running_tasks[investigation_id] = task

    base = f"/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}"
    return {
        "investigation_id": investigation_id,
        "status": "started",
        "item_name": item_name,
        "poll_url": base,                 # single polling endpoint:
                                          # running -> progress, completed -> full analysis
        "status_url": f"{base}/status",
        "analysis_url": f"{base}/analysis",
    }


@app.get("/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}")
async def investigation_poll(sub: str, service: str, investigation_id: str):
    try:
        manifest = await svc.investigation_status(sub, service, investigation_id)
    except PersistenceError as e:
        if investigation_id in _running_tasks:
            return {"investigation_id": investigation_id, "status": "starting"}
        raise _http(e)
    except Exception as e:
        raise _http(e)

    status = manifest.get("status")
    if status == "completed":
        try:
            analysis = await svc.compose_analysis(sub, service, investigation_id)
        except Exception as e:
            raise _http(e)
        return {"investigation_id": investigation_id, "status": "completed",
                "analysis": analysis}

    if status == "failed":
        return {"investigation_id": investigation_id, "status": "failed",
                "error": manifest.get("error"), "manifest": manifest}

    return {
        "investigation_id": investigation_id,
        "status": "running",
        "current_step": (manifest.get("stage_timings") or {}) and
                        sorted(manifest["stage_timings"])[-1] or "starting",
        "elapsed_seconds": manifest.get("total_seconds_so_far"),
        "dispatched_agents": manifest.get("dispatched_agents", []),
        "agent_runs": manifest.get("agent_runs"),
        "findings_count": manifest.get("findings_count"),
        "validated_findings_count": manifest.get("validated_findings_count"),
        "root_causes_count": manifest.get("root_causes_count"),
        "recommendations_count": manifest.get("recommendations_count"),
    }


@app.get("/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}/status")
async def investigation_status(sub: str, service: str, investigation_id: str):
    """Poll target: the persisted manifest (flushed at every checkpoint)."""
    try:
        manifest = await svc.investigation_status(sub, service, investigation_id)
    except PersistenceError as e:
        # id was just issued but first checkpoint not yet flushed
        if investigation_id in _running_tasks:
            return {"investigation_id": investigation_id, "status": "starting"}
        raise _http(e)
    except Exception as e:
        raise _http(e)
    manifest["in_this_process"] = investigation_id in _running_tasks
    return manifest


@app.get("/api/subscriptions/{sub}/services/{service}/investigations/{investigation_id}/analysis")
async def investigation_analysis(sub: str, service: str, investigation_id: str):
    """The full analysis screen, composed only from persisted files."""
    try:
        return await svc.compose_analysis(sub, service, investigation_id)
    except Exception as e:
        raise _http(e)


@app.get("/api/subscriptions/{sub}/services/{service}/rg/{rg}/ws/{ws}/items/{item_type}/{item_name}/investigations")
async def item_investigations(sub: str, service: str, rg: str, ws: str,
                              item_type: str, item_name: str,
                              limit: int = Query(default=10, le=50)):
    try:
        return await svc.investigation_history(sub, service, ws, item_type, item_name, limit)
    except Exception as e:
        raise _http(e)


@app.get("/api/health")
def health():
    return {"status": "ok",
            "resolver_version": RESOLVER_VERSION,
            "running_investigations": sorted(_running_tasks)}