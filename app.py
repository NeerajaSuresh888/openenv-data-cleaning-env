from __future__ import annotations
import uuid             #generate unique id for each task
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException      #HTTPException is used to handle errors like 404.
from pydantic import BaseModel

from env import DataCleaningEnv, Action


app = FastAPI(
    title = "Data Cleaning Environment API",
    description="OpenEnv-compliant Data Cleaning & Validation Environment ",
    version="1.0.0",
)


_sessions : Dict[str, DataCleaningEnv] ={}


class ResetRequest(BaseModel):      #what you send to start a new episode
    task_id: Optional[str] = None
    seed: int= 42


class StepRequest(BaseModel):       #what you send to take an action
    session_id: str
    action: Dict[str, Any]


class GradeRequest(BaseModel):     #what do you send to final score
    session_id: str

@app.get("/")
def root():
    return{"name": "Data Cleaning Environment API", "version": "1.0.0"}


@app.get("/health")
def health():
    return{"status":"ok"}


@app.post("/reset")
def reset(req: ResetRequest = Body(default=ResetRequest())):
    session_id = str(uuid.uuid4())
    env = DataCleaningEnv(task_id=req.task_id,seed=req.seed)
    result =env.reset(task_id = req.task_id)
    _sessions[session_id] = env
    return {"session_id": session_id, **result.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action format: {e}")
    result = env.step(action)
    return result.model_dump()


@app.get("/state/{session_id}")
def state(session_id :str):
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")
    return env.state().model_dump()



@app.post("/grade")
def grade(req: GradeRequest):
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")
    return{"session_id":req.session_id,"score":env.grade()}