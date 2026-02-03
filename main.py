import base64
import json
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/")
async def pubsub_push(request: Request):
    body = await request.json()

    # Pub/Sub push 포맷: {"message": {"data": "<base64>" ...}, "subscription": "..."}
    msg = body.get("message", {})
    data_b64 = msg.get("data", "")

    if not data_b64:
        # 테스트용: 그냥 원문을 로그로 확인
        return {"received": body}

    raw = base64.b64decode(data_b64).decode("utf-8")
    payload = json.loads(raw) if raw and raw.strip().startswith("{") else {"raw": raw}

    return {"ok": True, "payload": payload}
