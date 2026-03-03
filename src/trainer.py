"""
Training engine for GPT-2 with RoPE & GQA.

Features:
  - Cosine LR schedule with linear warmup
  - Mixed precision training (torch.amp)
  - Gradient clipping
  - Selective weight decay (only 2D parameters)
  - Periodic checkpointing + best model tracking
  - Optional Weights & Biases logging
  - Resume from checkpoint

Usage (standalone):
    python -m src.trainer --config configs/small.yaml
"""

import math
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_schedule(
    step:         int,
    warmup_steps: int,
    total_steps:  int,
    max_lr:       float,
    min_lr:       float,
) -> float:
    """
    Linear warmup followed by cosine decay.

    Phase 1 (0 → warmup_steps): lr linearly increases from 0 to max_lr
    Phase 2 (warmup → total):   lr cosine-decays from max_lr to min_lr
    """
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training loop for GPT2RoPEGQA.

    Args:
        model         : GPT2RoPEGQA instance
        train_loader  : DataLoader for training set
        val_loader    : DataLoader for validation set
        config        : dict with training hyperparameters (see defaults below)
        checkpoint_dir: directory to save .pt checkpoints
        use_wandb     : whether to log to Weights & Biases
    """

    DEFAULTS = dict(
        max_epochs    = 5,
        max_lr        = 6e-4,
        min_lr        = 6e-5,
        warmup_steps  = 2_000,
        weight_decay  = 0.1,
        beta1         = 0.9,
        beta2         = 0.95,
        eps           = 1e-8,
        grad_clip     = 1.0,
        eval_interval = 500,        # steps
        log_interval  = 50,
        save_interval = 1_000,
        dtype         = 'bfloat16', # 'float32', 'float16', 'bfloat16'
    )

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        config:         Optional[Dict[str, Any]] = None,
        checkpoint_dir: str  = 'checkpoints',
        use_wandb:      bool = False,
    ):
        self.model    = model
        self.train_dl = train_loader
        self.val_dl   = val_loader
        self.cfg      = {**self.DEFAULTS, **(config or {})}
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.device = next(model.parameters()).device

        # Mixed precision
        self.dtype = dict(
            float32=torch.float32,
            float16=torch.float16,
            bfloat16=torch.bfloat16,
        )[self.cfg['dtype']]
        self.scaler = GradScaler(enabled=(self.cfg['dtype'] == 'float16'))

        # Optimiser — separate param groups for weight decay
        self.optimizer = self._build_optimizer()

        # Training state
        self.step        = 0
        self.best_val_ppl = float('inf')
        self.train_losses: list = []
        self.val_losses:   list = []
        self.val_steps:    list = []

        # Total steps for LR schedule
        self.total_steps = self.cfg['max_epochs'] * len(train_loader)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        AdamW with weight decay only on 2D parameters (weight matrices).
        Embeddings, biases, LayerNorm params are not decayed.
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay.append(param)
            else:
                no_decay.append(param)

        groups = [
            {'params': decay,    'weight_decay': self.cfg['weight_decay']},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr    = self.cfg['max_lr'],
            betas = (self.cfg['beta1'], self.cfg['beta2']),
            eps   = self.cfg['eps'],
        )

    # ------------------------------------------------------------------
    # LR
    # ------------------------------------------------------------------

    def _update_lr(self) -> float:
        lr = cosine_schedule(
            self.step,
            warmup_steps = self.cfg['warmup_steps'],
            total_steps  = self.total_steps,
            max_lr       = self.cfg['max_lr'],
            min_lr       = self.cfg['min_lr'],
        )
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run full validation pass, return average loss."""
        self.model.eval()
        total, count = 0.0, 0
        for x, y in self.val_dl:
            x, y = x.to(self.device), y.to(self.device)
            with autocast(dtype=self.dtype):
                _, loss = self.model(x, y)
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        path = self.ckpt_dir / f"gpt2_rope_gqa_{tag}.pt"
        torch.save({
            'step':           self.step,
            'model_state':    self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_ppl':   self.best_val_ppl,
            'train_losses':   self.train_losses,
            'val_losses':     self.val_losses,
            'val_steps':      self.val_steps,
            'config':         self.cfg,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.step         = ckpt['step']
        self.best_val_ppl = ckpt.get('best_val_ppl', float('inf'))
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses   = ckpt.get('val_losses', [])
        self.val_steps    = ckpt.get('val_steps', [])
        print(f"Resumed from step {self.step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """
        Run full training.

        Returns:
            dict with 'train_losses', 'val_losses', 'val_steps'
        """
        self.model.train()
        t0 = time.time()

        for epoch in range(1, self.cfg['max_epochs'] + 1):
            for x, y in self.train_dl:
                x, y = x.to(self.device), y.to(self.device)

                # ── LR schedule ────────────────────────────────────────
                lr = self._update_lr()

                # ── Forward ────────────────────────────────────────────
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(dtype=self.dtype):
                    _, loss = self.model(x, y)

                # ── Backward ───────────────────────────────────────────
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.step += 1
                self.train_losses.append(loss.item())

                # ── Logging ────────────────────────────────────────────
                if self.step % self.cfg['log_interval'] == 0:
                    elapsed = time.time() - t0
                    ppl = math.exp(min(loss.item(), 20))
                    print(
                        f"Epoch {epoch:2d} | Step {self.step:6d} | "
                        f"loss={loss.item():.4f} | ppl={ppl:.2f} | "
                        f"lr={lr:.2e} | {elapsed:.0f}s elapsed"
                    )
                    t0 = time.time()

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/ppl':  ppl,
                            'train/lr':   lr,
                            'step':       self.step,
                        })

                # ── Validation ─────────────────────────────────────────
                if self.step % self.cfg['eval_interval'] == 0:
                    val_loss = self.evaluate()
                    val_ppl  = math.exp(min(val_loss, 20))
                    self.val_losses.append(val_loss)
                    self.val_steps.append(self.step)
                    print(
                        f"  ▶ Val: loss={val_loss:.4f} | ppl={val_ppl:.2f} "
                        f"{'  ← best!' if val_ppl < self.best_val_ppl else ''}"
                    )

                    if val_ppl < self.best_val_ppl:
                        self.best_val_ppl = val_ppl
                        self._save_checkpoint('best')

                    if self.use_wandb:
                        import wandb
                        wandb.log({'val/loss': val_loss, 'val/ppl': val_ppl, 'step': self.step})

                # ── Periodic checkpoint ────────────────────────────────
                if self.step % self.cfg['save_interval'] == 0:
                    self._save_checkpoint(f'step_{self.step}')

            # End of epoch checkpoint
            self._save_checkpoint(f'epoch_{epoch}')

        print(f"\nTraining complete. Best val PPL: {self.best_val_ppl:.2f}")
        return {
            'train_losses': self.train_losses,
            'val_losses':   self.val_losses,
            'val_steps':    self.val_steps,
        }
