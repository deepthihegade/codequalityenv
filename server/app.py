from fastapi import FastAPI, WebSocket
from environment import Environment
from models import Action
import json

app = FastAPI()
env = Environment()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        message = json.loads(data)

        if message["method"] == "reset":
            result = env.reset()
            await websocket.send_text(json.dumps({
                "observation": {
                    "buggy_code": result.observation.buggy_code,
                    "task_level": result.observation.task_level,
                    "error_hint": result.observation.error_hint
                },
                "reward": result.reward,
                "done": result.done,
                "feedback": result.feedback
            }))

        elif message["method"] == "step":
            action = Action(
                action_type=message["action"]["action_type"],
                line_number=message["action"].get("line_number"),
                code_patch=message["action"].get("code_patch")
            )
            result = env.step(action)
            await websocket.send_text(json.dumps({
                "observation": {
                    "buggy_code": result.observation.buggy_code,
                    "task_level": result.observation.task_level,
                    "error_hint": result.observation.error_hint
                },
                "reward": result.reward,
                "done": result.done,
                "feedback": result.feedback
            }))

        elif message["method"] == "state":
            state = env.state
            await websocket.send_text(json.dumps({
                "episode_id": state.episode_id,
                "step_count": state.step_count,
                "current_task": state.current_task
            }))