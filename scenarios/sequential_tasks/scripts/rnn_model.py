"""Event RNN (v2) – hidden == predicted h
===========================================================
This revision unifies the GRU hidden state and the target vector **hₜ** so
`prev_h` *is literally the GRU hidden*.  Concretely, the GRU hidden size
is now **h_dim = 1024**, and the projection layer `next_state_head` has
been dropped.

Public interface (constructor arguments, loss breakdown, training loop)
remains the same, but the internal tensor shapes are simpler:

```
prev_h ∈ ℝ^{B×h_dim}
gru_hidden ≡ prev_h
```

So the autoregressive update is now just

```python
gru_hidden = rnn_cell(fused_t, gru_hidden)  # new hidden == new hₜ
next_h     = gru_hidden
```

The rest of the file is a complete, runnable script.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

###############################################################################
# Constants – unchanged
###############################################################################

MAX_SEQ_LEN = 8

###############################################################################
# Dataset + DataLoader helpers – identical to v1 (see comments there)
###############################################################################

def make_loaders(
    json_path: str,
    batch_size: int = 128,
    train_frac: float = 0.8,
    *,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    full_ds = EventSequenceDataset.from_json(json_path)
    n_total = len(full_ds)
    n_train = math.floor(n_total * train_frac)
    n_val = n_total - n_train

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, (n_train, n_val), generator=gen)

    kwargs = dict(
        batch_size=batch_size,
        collate_fn=EventSequenceDataset.collate,
        num_workers=0,
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=True, **kwargs),
    )


class EventSequenceDataset(Dataset):
    """Parses raw JSON into tensor dicts."""

    def __init__(self, raw: List[Dict]):
        self.samples: List[Dict[str, Tensor]] = []
        for ix, s in enumerate(raw):
            e_key = "events" if "events" in s else "e"
            if e_key not in s or not all(k in s for k in ("y", "h", "success")):
                raise KeyError(f"Sample {ix} missing required keys")

            e = torch.tensor(s[e_key], dtype=torch.float32)
            h = torch.tensor(s["h"], dtype=torch.float32)
            y = torch.tensor(s["y"], dtype=torch.float32)
            state_label = torch.tensor(s["label"], dtype=torch.float32)
            self.samples.append({"e": e, "y": y, "h": h, "label": state_label})

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "EventSequenceDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError("JSON must be an array of samples")
        return cls(raw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    @staticmethod
    def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        lengths = torch.tensor([s["e"].size(0) for s in batch])
        max_len = MAX_SEQ_LEN
        event_dim = batch[0]["e"].size(1)
        h_dim = batch[0]["h"].size(1)
        label_dim = batch[0]["label"].size(1)

        e_out, h_out, y_out, labels = [], [], [], []
        for s in batch:
            T = s["e"].size(0)
            pad_e = torch.zeros((max_len - T, event_dim))
            pad_h = torch.zeros((max_len - T, h_dim))
            pad_label = torch.zeros((max_len - T,label_dim))

            e_out.append(torch.cat([s["e"], pad_e]))
            h_out.append(torch.cat([s["h"], pad_h]))
            y_rep = s["y"].unsqueeze(0).expand(max_len, -1)
            y_out.append(y_rep)
            labels.append(torch.cat([s["label"], pad_label]))

        return {
            "e": torch.stack(e_out),
            "h": torch.stack(h_out),
            "y": torch.stack(y_out),
            "label": torch.stack(labels),
            "lengths": lengths
        }

###############################################################################
# EventRNN – hidden == h
###############################################################################

class EventRNN(pl.LightningModule):
    """RNN variant where the GRU hidden state *is* the predicted hₜ."""

    def __init__(
        self,
        *,
        event_dim: int,
        y_dim: int,
        latent_dim: int,
        state_dim: int,
        input_dim: int = 16,
        num_layers: int = 1,
        lr: float = 1e-4,
        cls_loss_weight: float = 1.0,
        recon_loss: str = "cosine",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        D = latent_dim  # hidden size now equals target size
        I = input_dim

        # ─── Embeddings ──────────────────────────────────────────────
        self.e_proj = nn.Linear(event_dim, I)
        self.gamma = nn.Linear(y_dim, I)
        self.beta = nn.Linear(y_dim, I)
        hidden_dim = max(D // 2, 4)
        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, I),
        )
        # self.h_proj = nn.Identity()  # no change in dimensionality
        # self.fuse = nn.Linear(2 * I, I)

        # ─── GRU stack ───────────────────────────────────────────────
        if num_layers == 1:
            self.rnn = nn.GRUCell(2*I, D)
        else:
            self.rnn = nn.GRU(I, D, num_layers=num_layers, batch_first=True)
            # we will still step manually to keep parity with TBPTT logic

        # ─── Classification head ────────────────────────────────────
        hidden_dim = max(D // 2, 4)
        self.state_head = nn.Sequential(
            nn.Linear(D + 2*I, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, D),
        )

        self.recon_loss = recon_loss.lower()
        if self.recon_loss not in {"mse", "cosine"}:
            raise ValueError("recon_loss must be 'mse' or 'cosine'")
        self.bce = nn.BCEWithLogitsLoss()
        self.cls_w = cls_loss_weight
        self.lr = lr

    # ------------------------------------------------------------------
    def _prepare_step(self, e: Tensor, y: Tensor) -> Tensor:
        emb_e = self.e_proj(e)  # (B, T, I)          
        gamma = self.gamma(y)
        beta = self.beta(y)
        return gamma * emb_e + beta 


    # ------------------------------------------------------------------
    def _rollout(self, e: Tensor, y: Tensor, lengths: Tensor) -> Tensor:
        B, T, _ = e.shape
        D = self.hparams.latent_dim
        L = self.hparams.num_layers


        fused = torch.cat([self.e_proj(e),self.y_proj(y)], dim=-1)  # (B, T, 2*I)

        # ───── Manual stepping for GRUCell ─────
        if isinstance(self.rnn, nn.GRUCell):
            h = e.new_zeros(B, D)
            preds = []
            for t in range(T):
                h = self.rnn(fused[:, t], h)
                preds.append(h.clone())
            out = torch.stack(preds, dim=1)  # (B, T, D)

        # ───── Full-sequence GRU ─────
        else:
            h0 = e.new_zeros(L, B, D)
            out, _ = self.rnn(fused, h0)  # (B, T, D)

        # ───── Mask out padded positions ─────
        mask = (
            torch.arange(T, device=lengths.device)
            .unsqueeze(0)                # (1, T)
            .expand(B, T)                # (B, T)
            < lengths.unsqueeze(1)       # (B, 1)
        )                                # → bool mask of shape (B, T)

        out = out * mask.unsqueeze(-1)   # (B, T, D)
        
        state_decoder_out = self.state_head(torch.cat([out,fused], dim=-1))  # (B, T, state_dim)
        recon_out = self.recon_head(out)  # (B, T, state_dim)
        #decoder_out = self.cls_head(out)  # (B, T, state_dim)

        return state_decoder_out, recon_out

    def forward(self, e: Tensor, y: Tensor, lengths: Tensor) -> Tensor:
        """Single forward pass through the RNN.
        Args:
            e: Event embedding of shape (B, 1, e_dim)
            y: Y embedding of shape (B, 1, y_dim)
            lengths: Sequence lengths of shape (B,)
        Returns:
            recon_out: Reconstruction output of shape (B, 1, D)
        """
        
        # Prepare the input embeddings
        fused = torch.cat([self.e_proj(e),self.y_proj(y)], dim=-1)  # (B, T, 2*I)
        h = self.rnn(fused)  # (B, T, D)
        # Apply reconstruction head
        recon_out = self.recon_head(h)  
        
        return recon_out

    # ------------------------------------------------------------------
    def _step(self, batch):
        e, y, target_h = batch["e"], batch["y"], batch["h"] # (B, T, e_dim), (B, T, y_dim), (B, T+1, D)
        lengths = batch["lengths"] # (B,)
        label = batch["label"] # (B,T+1, state_dim)

        pred_cls, pred_recon = self._rollout(e, y, lengths)
        
        # Only compute reconstruction loss for positive labels      
        target = target_h[:, 1:]  # (B, T, D)

        # Mask: (B, T)
        mask = (
            torch.arange(target.size(1), device=lengths.device)
            .unsqueeze(0).expand(lengths.size(0), -1)
            < lengths.unsqueeze(1)
        )

        if self.recon_loss == "mse":
            # (B, T, D) -> (B, T)
            mse = F.mse_loss(pred_recon, target, reduction='none').mean(dim=-1)
            loss_recon = (mse * mask).sum() / mask.sum()
        else:
            cos = F.cosine_similarity(pred_recon, target, dim=-1)  # (B, T)
            loss_recon = ((1 - cos) * mask).sum() / mask.sum()

        # Compute BCE loss per element: (B, T, state_dim)
        raw_loss_cls = F.binary_cross_entropy_with_logits(pred_cls, label[:, 1:], reduction='none')

        # Apply mask: only count valid timesteps
        masked_loss = raw_loss_cls * mask.unsqueeze(-1)

        # Normalize by number of unmasked elements
        loss_cls = masked_loss.sum() / mask.sum()

        loss = loss_recon + self.cls_w * loss_cls
        #loss = loss_cls
        return loss, loss_recon, loss_cls

    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, rec, cls = self._step(batch)
        self.log_dict({"train_loss": loss, "train_recon": rec, "train_cls": cls}, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, rec, cls = self._step(batch)
        self.log_dict({"val_loss": loss, "val_recon": rec, "val_cls": cls}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

###############################################################################
# CLI entrypoint
###############################################################################
if __name__ == "__main__":
    train_loader, val_loader = make_loaders(
        "sequential_tasks/data/dataset_full.json", batch_size=128
    )

    model = EventRNN(event_dim=3, y_dim=1024, latent_dim=1024, state_dim=4, input_dim=16, num_layers=1)

    trainer = pl.Trainer(max_epochs=400, accelerator="auto", log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
