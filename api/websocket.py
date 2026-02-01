from fastapi import WebSocket, WebSocketDisconnect
from core.training_manager import TrainingManager
import asyncio
import json


async def handle_websocket(websocket: WebSocket):
    """Handle WebSocket connection for real-time updates"""
    await websocket.accept()

    manager = TrainingManager()

    # Set the event loop for async broadcasting from background threads
    if manager.loop is None:
        manager.loop = asyncio.get_event_loop()

    manager.add_websocket(websocket)

    try:
        # Send initial status
        status = manager.get_status()
        await websocket.send_json({
            "type": "initial_status",
            "data": status
        })

        # Keep connection alive and handle any client messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle client messages if needed (e.g., ping/pong)
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.remove_websocket(websocket)
