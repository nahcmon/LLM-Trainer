import torch
from tqdm import tqdm
from dataset import load_hf_dataset, load_multiple_datasets
from tokenizer import train_tokenizer
from model import build_scratch_model
from model_export import export_model
import time

class Trainer:
    def __init__(self, cfg, progress_callback=None, log_callback=None):
        """
        Args:
            cfg: Configuration dictionary with all training parameters
            progress_callback: Callable that receives progress updates
                Signature: callback(step, total_steps, metrics_dict)
            log_callback: Callable that receives log messages
                Signature: callback(level, message)
        """
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.should_stop = False
        self.is_training = False

        self._log("info", f"Initializing model on device: {self.device}")
        self.model = build_scratch_model(cfg, precision=cfg.get("precision", "fp16"))
        self.model.to(self.device)

    def train(self):
        """Main training loop with progress tracking"""
        self.is_training = True
        self.should_stop = False

        try:
            # Setup phase
            self._log("info", "Loading dataset...")
            ds_start_time = time.time()

            # Check if multiple datasets are specified
            dataset_spec = self.cfg["dataset"]
            if ',' in dataset_spec:
                # Multiple datasets
                self._log("info", "Multiple datasets detected, loading and combining...")
                ds = load_multiple_datasets(
                    dataset_spec,
                    split=self.cfg.get("split", "train"),
                    mixing_strategy=self.cfg.get("dataset_mixing", "interleave")
                )
            else:
                # Single dataset
                ds = load_hf_dataset(dataset_spec, split=self.cfg.get("split", "train"))

            ds_load_time = time.time() - ds_start_time
            self._log("info", f"Dataset loaded: {len(ds)} examples in {ds_load_time:.2f}s")

            # Prepare tokenizer
            self._log("info", "Training tokenizer...")
            tok_start_time = time.time()
            self._log("info", "  → Extracting text from dataset...")
            dataset_size = len(ds)
            with open("tmp_text.txt", "w", encoding="utf-8") as f:
                for i, ex in enumerate(ds):
                    if i % 1000 == 0:
                        # Report progress during text extraction
                        progress = (i + 1) / dataset_size
                        self._report_progress(i + 1, dataset_size, {
                            "epoch": 0,
                            "loss": 0.0,
                            "learning_rate": 0.0,
                            "phase": "tokenizer_training"
                        })
                        self._log("info", f"  → Processed {i}/{dataset_size} examples ({100*i/dataset_size:.1f}%)")
                    f.write(ex.get("text", "") + "\n")
            # Final progress update for text extraction
            self._report_progress(dataset_size, dataset_size, {
                "epoch": 0,
                "loss": 0.0,
                "learning_rate": 0.0,
                "phase": "tokenizer_training_complete"
            })
            self._log("info", f"  → Text extraction complete")
            tokenizer_path = train_tokenizer(["tmp_text.txt"], vocab_size=self.cfg["vocab_size"])
            tok_time = time.time() - tok_start_time
            self._log("info", f"Tokenizer trained at {tokenizer_path} in {tok_time:.2f}s")

            # Load the trained tokenizer and tokenize dataset
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            self._log("info", "Tokenizing dataset...")

            tokenized_examples = []
            seq_length = self.cfg["seq_length"]
            for i, ex in enumerate(ds):
                if i % 1000 == 0:
                    self._log("info", f"  → Tokenized {i}/{len(ds)} examples")
                text = ex.get("text", "")
                token_ids = sp.encode(text, out_type=int)
                # Truncate or pad to seq_length
                if len(token_ids) > seq_length:
                    token_ids = token_ids[:seq_length]
                else:
                    token_ids = token_ids + [0] * (seq_length - len(token_ids))
                tokenized_examples.append({"input_ids": token_ids})

            self._log("info", f"Tokenization complete: {len(tokenized_examples)} examples")
            # Replace dataset with tokenized version
            ds = tokenized_examples

            # Setup optimizer with configurable parameters
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.get("learning_rate", 3e-4),
                weight_decay=self.cfg.get("weight_decay", 0.01),
                betas=(self.cfg.get("adam_beta1", 0.9), self.cfg.get("adam_beta2", 0.999)),
                eps=self.cfg.get("adam_epsilon", 1e-8)
            )

            # Setup gradient scaler for automatic mixed precision (AMP)
            use_amp = self.cfg.get("precision", "fp16") in ["fp16", "bf16"] and self.device == "cuda"
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            self._log("info", f"Automatic Mixed Precision: {'ENABLED' if use_amp else 'DISABLED'}")

            # Calculate total steps
            total_steps = self._calculate_total_steps(len(ds))
            current_step = 0
            training_start_time = time.time()
            step_times = []

            self._log("info", f"Starting training for {self.cfg['epochs']} epoch(s), {total_steps} total steps")

            # Initialize progress bar at 0% for training phase
            self._report_progress(0, total_steps, {
                "epoch": 0,
                "loss": 0.0,
                "learning_rate": self.cfg.get("learning_rate", 3e-4)
            })

            # Training loop
            for epoch in range(self.cfg["epochs"]):
                if self.should_stop:
                    self._log("warning", "Training stopped by user")
                    break

                self._log("info", f"Epoch {epoch + 1}/{self.cfg['epochs']}")
                epoch_start_time = time.time()

                for batch_idx, ex in enumerate(ds):
                    if self.should_stop:
                        self._log("warning", "Training stopped by user")
                        break

                    # Training step with timing
                    step_start = time.time()
                    loss = self._training_step(ex, optimizer, scaler)
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    current_step += 1

                    # Calculate speed metrics
                    if len(step_times) > 100:
                        step_times = step_times[-100:]  # Keep last 100 for rolling average

                    # Report progress
                    if current_step % self.cfg.get("logging_steps", 10) == 0:
                        avg_step_time = sum(step_times) / len(step_times)
                        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                        elapsed_time = time.time() - training_start_time
                        remaining_steps = total_steps - current_step
                        eta_seconds = remaining_steps * avg_step_time

                        metrics = {
                            "epoch": epoch + 1,
                            "loss": loss,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "step_time": avg_step_time,
                            "steps_per_sec": steps_per_sec,
                            "elapsed_time": elapsed_time,
                            "eta_seconds": eta_seconds,
                        }
                        self._report_progress(current_step, total_steps, metrics)

                        # Log progress with speed metrics
                        eta_str = self._format_time(eta_seconds)
                        elapsed_str = self._format_time(elapsed_time)
                        self._log("info",
                            f"Step {current_step}/{total_steps} | "
                            f"Loss: {loss:.4f} | "
                            f"Speed: {steps_per_sec:.2f} steps/s | "
                            f"Elapsed: {elapsed_str} | "
                            f"ETA: {eta_str}")

                    # Check if max_steps reached
                    max_steps = self.cfg.get("max_steps")
                    if max_steps and current_step >= max_steps:
                        self._log("info", f"Reached max_steps ({max_steps})")
                        self.should_stop = True
                        break

                epoch_time = time.time() - epoch_start_time
                steps_in_epoch = min(len(ds), total_steps - (current_step - batch_idx - 1))
                examples_per_sec = steps_in_epoch / epoch_time if epoch_time > 0 else 0
                self._log("info",
                    f"Epoch {epoch + 1} completed in {epoch_time:.2f}s "
                    f"({examples_per_sec:.2f} examples/s)")

            if not self.should_stop:
                self._log("info", "Training completed successfully!")

                # Export model after training
                export_format = self.cfg.get("export_format", "safetensors")
                self._log("info", f"Exporting model in {export_format} format...")
                try:
                    output_path = export_model(
                        self.model,
                        tokenizer_path,
                        output_dir=self.cfg.get("output_dir", "output"),
                        export_format=export_format
                    )
                    self._log("info", f"Model exported successfully to: {output_path}")
                except Exception as e:
                    self._log("error", f"Failed to export model: {str(e)}")

        finally:
            self.is_training = False

    def _training_step(self, example, optimizer, scaler):
        """Execute a single training step with automatic mixed precision"""
        # Prepare input - input_ids must ALWAYS be Long integers for embedding lookup
        ids = torch.tensor(
            [example.get("input_ids", [0] * self.cfg["seq_length"])],
            dtype=torch.long  # Must be Long for embedding layer
        ).to(self.device)

        # Determine autocast settings
        precision = self.cfg.get("precision", "fp16")
        use_autocast = precision in ["fp16", "bf16"] and self.device == "cuda"
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # Forward pass with autocast
        with torch.amp.autocast('cuda', enabled=use_autocast, dtype=dtype):
            outputs = self.model(ids, labels=ids)
            loss = outputs.loss

        # Check for NaN loss
        if torch.isnan(loss):
            self._log("warning", "NaN loss detected, skipping step")
            optimizer.zero_grad(set_to_none=True)
            return 0.0

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)

        # Gradient clipping
        max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Clear CUDA cache periodically to prevent memory fragmentation
        # Only do this every N steps to avoid performance overhead
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
        else:
            self._step_counter = 1

        if self._step_counter % 100 == 0 and self.device == "cuda":
            torch.cuda.empty_cache()

        return loss.item()

    def _calculate_total_steps(self, dataset_size):
        """Calculate total number of training steps"""
        max_steps = self.cfg.get("max_steps")
        if max_steps:
            return max_steps

        epochs = self.cfg.get("epochs", 1)
        # Since the training loop processes one example at a time,
        # total steps = dataset_size * epochs (not divided by batch_size)
        return dataset_size * epochs

    def stop(self):
        """Request graceful shutdown"""
        self.should_stop = True
        self._log("warning", "Stop requested, finishing current step...")

    def _report_progress(self, step, total_steps, metrics):
        """Send progress update to callback"""
        if self.progress_callback:
            self.progress_callback(step, total_steps, metrics)

    def _log(self, level, message):
        """Send log message to callback"""
        if self.log_callback:
            self.log_callback(level, message)
        print(f"[{level.upper()}] {message}")

    def _format_time(self, seconds):
        """Format seconds into human-readable time string"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"
