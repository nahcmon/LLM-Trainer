import asyncio
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from trainer import Trainer


class TrainingState(Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


class TrainingManager:
    """Singleton manager for training operations"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.state = TrainingState.IDLE
        self.trainer: Optional[Trainer] = None
        self.training_thread: Optional[threading.Thread] = None
        self.current_config: Optional[Dict[str, Any]] = None
        self.progress_data = {
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "loss": 0.0,
            "metrics": {}
        }
        self.logs = []
        self.start_time: Optional[datetime] = None
        self.error_message: Optional[str] = None

        # WebSocket connections for broadcasting
        self.websocket_connections = set()

        # Event loop for async operations
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self._initialized = True

    def start_training(self, config: Dict[str, Any]):
        """Start training in background thread"""
        if self.state in [TrainingState.RUNNING, TrainingState.STARTING]:
            raise ValueError("Training is already running")

        self.state = TrainingState.STARTING
        self.current_config = config
        self.start_time = datetime.now()
        self.error_message = None
        self.logs = []
        self.progress_data = {
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "loss": 0.0,
            "metrics": {}
        }

        # Create trainer with callbacks
        self.trainer = Trainer(
            cfg=config,
            progress_callback=self._on_progress,
            log_callback=self._on_log
        )

        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            daemon=True
        )
        self.training_thread.start()

    def _schedule_async(self, coro):
        """Safely schedule async coroutine from background thread"""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        # If no loop is available, silently skip (no WebSocket clients connected)

    def _run_training(self):
        """Execute training (runs in background thread)"""
        try:
            self.state = TrainingState.RUNNING
            self._schedule_async(self._broadcast_state_change())

            self.trainer.train()

            self.state = TrainingState.COMPLETED
            self._on_log("info", "Training completed successfully")

        except Exception as e:
            self.state = TrainingState.ERROR
            self.error_message = str(e)
            self._on_log("error", f"Training failed: {e}")

        finally:
            self._schedule_async(self._broadcast_state_change())

    def stop_training(self):
        """Request graceful training stop"""
        if self.state != TrainingState.RUNNING:
            raise ValueError("No training is currently running")

        self.state = TrainingState.STOPPING
        self._on_log("warning", "Stopping training...")

        if self.trainer:
            self.trainer.stop()

    def _on_progress(self, step: int, total_steps: int, metrics: Dict):
        """Handle progress update from trainer"""
        # Sanitize metrics to ensure JSON compatibility (replace NaN/Inf with None)
        import math
        sanitized_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    sanitized_metrics[key] = 0.0
                else:
                    sanitized_metrics[key] = value
            else:
                sanitized_metrics[key] = value

        loss_value = sanitized_metrics.get("loss", 0.0)
        if isinstance(loss_value, float) and (math.isnan(loss_value) or math.isinf(loss_value)):
            loss_value = 0.0

        self.progress_data.update({
            "current_step": step,
            "total_steps": total_steps,
            "current_epoch": sanitized_metrics.get("epoch", 0),
            "loss": loss_value,
            "metrics": sanitized_metrics,
            "elapsed_time": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        })

        # Broadcast to all WebSocket connections
        self._schedule_async(self._broadcast_progress())

    def _on_log(self, level: str, message: str):
        """Handle log message from trainer"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)

        # Keep only last 1000 log entries
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]

        # Broadcast to all WebSocket connections
        self._schedule_async(self._broadcast_log(log_entry))

    async def _broadcast_progress(self):
        """Broadcast progress to all connected WebSocket clients"""
        if not self.websocket_connections:
            return

        message = {
            "type": "progress",
            "data": self.progress_data
        }

        # Send to all connections
        disconnected = set()
        for ws in self.websocket_connections.copy():
            try:
                await ws.send_json(message)
            except:
                disconnected.add(ws)

        # Clean up disconnected clients
        self.websocket_connections -= disconnected

    async def _broadcast_log(self, log_entry: Dict):
        """Broadcast log entry to all connected WebSocket clients"""
        if not self.websocket_connections:
            return

        message = {
            "type": "log",
            "data": log_entry
        }

        disconnected = set()
        for ws in self.websocket_connections.copy():
            try:
                await ws.send_json(message)
            except:
                disconnected.add(ws)

        self.websocket_connections -= disconnected

    async def _broadcast_state_change(self):
        """Broadcast state change to all clients"""
        if not self.websocket_connections:
            return

        message = {
            "type": "state_change",
            "data": {
                "state": self.state.value,
                "error_message": self.error_message
            }
        }

        disconnected = set()
        for ws in self.websocket_connections.copy():
            try:
                await ws.send_json(message)
            except:
                disconnected.add(ws)

        self.websocket_connections -= disconnected

    def add_websocket(self, ws):
        """Register WebSocket connection"""
        self.websocket_connections.add(ws)

    def remove_websocket(self, ws):
        """Unregister WebSocket connection"""
        self.websocket_connections.discard(ws)

    def unload_model(self):
        """Unload model from VRAM to free up memory"""
        import torch
        import gc

        if self.trainer and self.trainer.model:
            # Delete model
            del self.trainer.model
            self.trainer.model = None

        if self.trainer:
            del self.trainer
            self.trainer = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        self._on_log("info", "Model unloaded from VRAM")

    def get_status(self) -> Dict:
        """Get current training status"""
        return {
            "state": self.state.value,
            "progress": self.progress_data,
            "config": self.current_config,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "error_message": self.error_message
        }
