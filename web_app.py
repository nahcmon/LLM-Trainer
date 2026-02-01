from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router
from api.websocket import handle_websocket
import uvicorn
import os

app = FastAPI(
    title="LLM Training Web Interface",
    description="Web interface for training language models from scratch",
    version="1.0.0"
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router, prefix="/api")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)


# Serve index.html at root
@app.get("/")
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return {"message": "LLM Training API is running. Please create static/index.html for the web interface."}


if __name__ == "__main__":
    print("Starting LLM Training Web Interface on http://localhost:2345")
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=2345,
        reload=True,
        log_level="info"
    )
