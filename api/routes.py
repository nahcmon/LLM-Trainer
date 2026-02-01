from fastapi import APIRouter, HTTPException
from api.models import TrainingConfigRequest, TrainingStatusResponse
from core.training_manager import TrainingManager
from utils.parameter_definitions import PARAMETER_DEFINITIONS, PARAMETER_CATEGORIES
from utils.resource_estimator import estimate_vram_usage, estimate_training_time
from utils.model_presets import calculate_model_params

router = APIRouter()


@router.get("/parameters")
async def get_parameters():
    """Get all parameter definitions for UI generation"""
    return {
        "categories": PARAMETER_CATEGORIES,
        "definitions": PARAMETER_DEFINITIONS
    }


@router.post("/training/start")
async def start_training(config: TrainingConfigRequest):
    """Start a new training run"""
    manager = TrainingManager()

    try:
        manager.start_training(config.dict())
        return {"status": "started", "message": "Training started successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/training/stop")
async def stop_training():
    """Stop the current training run"""
    manager = TrainingManager()

    try:
        manager.stop_training()
        return {"status": "stopping", "message": "Training stop requested"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")


@router.get("/training/status", response_model=TrainingStatusResponse)
async def get_status():
    """Get current training status"""
    manager = TrainingManager()
    return manager.get_status()


@router.get("/training/logs")
async def get_logs(limit: int = 100):
    """Get recent training logs"""
    manager = TrainingManager()
    return {"logs": manager.logs[-limit:]}


@router.post("/training/unload")
async def unload_model():
    """Unload model from VRAM"""
    manager = TrainingManager()

    try:
        manager.unload_model()
        return {"status": "unloaded", "message": "Model unloaded from VRAM successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@router.post("/training/estimate")
async def estimate_resources(config: TrainingConfigRequest):
    """Estimate VRAM and training time for given configuration"""
    try:
        # Calculate model parameters
        param_count = calculate_model_params(
            vocab_size=config.vocab_size,
            n_layers=config.n_layers,
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            seq_length=config.seq_length
        )

        # Estimate VRAM
        vram_estimate = estimate_vram_usage(
            vocab_size=config.vocab_size,
            n_layers=config.n_layers,
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            seq_length=config.seq_length,
            batch_size=config.batch_size,
            precision=config.precision,
            gradient_checkpointing=config.gradient_checkpointing
        )

        # Estimate training time (we need dataset size, which we don't have yet)
        # For now, use a rough estimate
        dataset_size = 10000  # placeholder
        time_estimate = estimate_training_time(
            dataset_size=dataset_size,
            epochs=config.epochs,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            model_params_millions=param_count / 1_000_000
        )

        return {
            "parameter_count": param_count,
            "vram": vram_estimate,
            "time": time_estimate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to estimate resources: {str(e)}")
