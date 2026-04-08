from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from environment import Environment
from models import Action

app = FastAPI()
env = Environment()

# Serve static files if you have CSS/JS assets
app.mount("/static", StaticFiles(directory="server"), name="static")

# Serve the frontend UI
@app.get("/")
def serve_ui():
    return FileResponse("server/index.html")


@app.post("/openenv/reset")
def reset():
    result = env.reset()
    return {
        "observation": {
            "buggy_code": result.observation.buggy_code,
            "task_level": result.observation.task_level,
            "error_hint": result.observation.error_hint
        },
        "reward": result.reward,
        "done": result.done,
        "feedback": result.feedback
    }


@app.post("/openenv/step")
def step(action: Action):
    result = env.step(action)
    return {
        "observation": {
            "buggy_code": result.observation.buggy_code,
            "task_level": result.observation.task_level,
            "error_hint": result.observation.error_hint
        },
        "reward": result.reward,
        "done": result.done,
        "feedback": result.feedback
    }


@app.get("/openenv/state")
def get_state():
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "current_task": s.current_task
    }


def main():
    """Entry point for the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

