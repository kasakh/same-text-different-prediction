#!/usr/bin/env python3
"""Fine-tune text classifiers for production label-stability experiments.

This is a robust replacement for the earlier project-specific training script.
It keeps model families close to the uploaded Gen-vs-Disc-style code while adding:
- config/status artifacts
- tokenizer and label mapping artifacts
- resume from last.ckpt
- raw logits preserved in eval outputs
- optional training batch size as a normal hyperparameter, not a core nondet axis
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pls.common import (
    apply_seed_profile,
    ensure_dir,
    environment_fingerprint,
    mark_success,
    sha256_file,
    sha256_json,
    sha256_tree,
    write_json_atomic,
)
from pls.data import get_dataset_source, load_train_val_data

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 2e-5
VAL_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
MLM_MASK_PROB = 0.15
TRAINER_PRECISION = 32
NUM_WORKERS = 4
CHECKPOINT_TOP_K = 3
BERT_VOCAB_SIZE = 30522
GPT2_VOCAB_SIZE = 50257
MAX_POSITION_EMBEDDINGS = 512
DEFAULT_PRETRAINED_GPT2_MODEL = "openai-community/gpt2"
DEFAULT_PRETRAINED_GPT2_REVISION = (
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

MODEL_CONFIGS = {
    "bert": {
        "small": {"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 1, "intermediate_size": 256},
        "med": {"hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6, "intermediate_size": 1536},
        "large": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
        "pretrained": {"hidden_size": 768},
    },
    "gpt2": {
        "small": {"n_embd": 64, "n_layer": 1, "n_head": 1},
        "med": {"n_embd": 384, "n_layer": 6, "n_head": 6},
        "large": {"n_embd": 768, "n_layer": 12, "n_head": 12},
        "pretrained": {"n_embd": 768},
    },
}

MODEL_TYPES = ["BERT", "pseudo-gen-MLM", "pure-gen-GPT2", "pseudo-gen-GPT2"]
MODEL_SIZES = ["small", "med", "large", "pretrained"]


def _is_immutable_hf_revision(value: Optional[str]) -> bool:
    return bool(
        value
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = MAX_POSITION_EMBEDDINGS, model_type: str = "BERT"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        self.label_tokens = {}
        if self.model_type in ["pseudo-gen-MLM", "pure-gen-GPT2", "pseudo-gen-GPT2"]:
            label_gen_id = self.tokenizer.convert_tokens_to_ids("[label_gen]")
            for label in sorted(set(self.labels)):
                label_id = self.tokenizer.convert_tokens_to_ids(f"[label_{label}]")
                self.label_tokens[label] = [label_gen_id, label_id]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        if self.model_type == "pseudo-gen-MLM":
            effective_max_length = self.max_length - 4
        elif self.model_type in ["pseudo-gen-GPT2", "pure-gen-GPT2"]:
            effective_max_length = self.max_length - 2
        elif self.model_type == "modern-decoder":
            effective_max_length = self.max_length
        else:
            effective_max_length = self.max_length

        if self.model_type == "pseudo-gen-MLM":
            enc = self.tokenizer(text, truncation=True, max_length=effective_max_length, add_special_tokens=False)
            input_ids = [self.tokenizer.cls_token_id] + enc["input_ids"] + self.label_tokens[label] + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_length:
                input_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)
        elif self.model_type == "pseudo-gen-GPT2":
            enc = self.tokenizer(text, truncation=True, max_length=effective_max_length, add_special_tokens=False)
            input_ids = enc["input_ids"] + self.label_tokens[label]
            attention_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_length:
                input_ids.insert(0, self.tokenizer.pad_token_id)
                attention_mask.insert(0, 0)
        elif self.model_type == "pure-gen-GPT2":
            enc = self.tokenizer(text, truncation=True, max_length=effective_max_length, add_special_tokens=False)
            input_ids = self.label_tokens[label] + enc["input_ids"]
            attention_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_length:
                input_ids.insert(0, self.tokenizer.pad_token_id)
                attention_mask.insert(0, 0)
        elif self.model_type == "modern-decoder":
            enc = self.tokenizer(text, truncation=True, max_length=effective_max_length, add_special_tokens=False)
            input_ids = list(enc["input_ids"])
            attention_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_length:
                input_ids.insert(0, self.tokenizer.pad_token_id)
                attention_mask.insert(0, 0)
        else:
            enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            input_ids = enc["input_ids"].flatten().tolist()
            attention_mask = enc["attention_mask"].flatten().tolist()
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TextDataModule(pl.LightningDataModule):
    def __init__(self, train_data: Tuple[List[str], List[int]], val_data: Tuple[List[str], List[int]], tokenizer, model_type: str, batch_size: int):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = TextDataset(self.train_data[0], self.train_data[1], self.tokenizer, model_type=self.model_type)
        self.val_dataset = TextDataset(self.val_data[0], self.val_data[1], self.tokenizer, model_type=self.model_type)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


class _MetricMixin:
    def _reset_epoch_cache(self):
        self.all_probs = []
        self.all_labels = []

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        probs = outputs.probs if hasattr(outputs, "probs") else torch.softmax(outputs.raw_logits, dim=-1)
        self.all_probs.append(probs.detach().cpu())
        self.all_labels.append(batch["labels"].detach().cpu())
        self.log("val_loss", outputs.loss, prog_bar=True)

    def on_validation_epoch_end(self):
        probs = torch.cat(self.all_probs)
        targets = torch.cat(self.all_labels)
        probs = self.all_gather(probs)
        targets = self.all_gather(targets)
        if probs.dim() > 2:
            probs = probs.view(-1, probs.size(-1))
        if targets.dim() > 1:
            targets = targets.view(-1)
        targets, probs = targets.cpu(), probs.cpu()
        preds = torch.argmax(probs, dim=-1)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        w_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
        try:
            auroc = roc_auc_score(targets, probs, multi_class="ovr", average="macro") if self.num_labels > 2 else roc_auc_score(targets, probs[:, 1])
        except Exception:
            auroc = 0.0
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_macro_f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_weighted_f1", w_f1, prog_bar=True, sync_dist=True)
        self.log("val_auroc", auroc, prog_bar=True, sync_dist=True)
        self._reset_epoch_cache()


class BertModule(_MetricMixin, pl.LightningModule):
    def __init__(self, model_size: str, labels: List[int], learning_rate: float = 2e-5, num_warmup_steps: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.labels = labels
        self.num_labels = len(labels)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self._reset_epoch_cache()
        if model_size == "pretrained":
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels)
        else:
            config = BertConfig(vocab_size=BERT_VOCAB_SIZE, max_position_embeddings=MAX_POSITION_EMBEDDINGS, num_labels=self.num_labels, **MODEL_CONFIGS["bert"][model_size])
            self.model = BertForSequenceClassification(config)

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, position_ids=position_ids)
        raw_logits = outputs.logits
        return type("ModelOutput", (), {"loss": outputs.loss, "raw_logits": raw_logits, "probs": torch.softmax(raw_logits, dim=-1)})()

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class MLMModule(_MetricMixin, pl.LightningModule):
    def __init__(self, model_size: str, model_type: str, labels: List[int], learning_rate: float = 2e-5, num_warmup_steps: int = 0, label_token_ids: Optional[List[int]] = None, vocab_size: int = BERT_VOCAB_SIZE):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.labels = labels
        self.num_labels = len(labels)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.label_token_ids = label_token_ids or []
        self._reset_epoch_cache()
        if model_size == "pretrained":
            self.bert = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        else:
            config = BertConfig(vocab_size=vocab_size, max_position_embeddings=MAX_POSITION_EMBEDDINGS, **MODEL_CONFIGS["bert"][model_size])
            self.bert = AutoModelForMaskedLM.from_config(config)
        if vocab_size > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(vocab_size)

    def _apply_masking_vectorized(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        masked_input_ids = input_ids.clone()
        mlm_labels = torch.full_like(input_ids, -100)
        last_real_positions = attention_mask.sum(dim=1) - 1
        exclude_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        exclude_mask[:, 0] = True
        exclude_mask[torch.arange(batch_size, device=device), last_real_positions] = True
        maskable_mask = attention_mask.bool() & ~exclude_mask
        should_mask = (torch.rand_like(maskable_mask.float()) < MLM_MASK_PROB) & maskable_mask
        mlm_labels[should_mask] = input_ids[should_mask]
        # Use mask token id from BERT tokenizer; fixed ID is 103.
        masked_input_ids[should_mask] = 103
        if self.model_type == "pseudo-gen-MLM":
            idx = torch.arange(batch_size, device=device)
            label_y_pos = torch.clamp(last_real_positions - 1, 0, seq_len - 1)
            label_gen_pos = torch.clamp(last_real_positions - 2, 0, seq_len - 1)
            mlm_labels[idx, label_y_pos] = input_ids[idx, label_y_pos]
            mlm_labels[idx, label_gen_pos] = input_ids[idx, label_gen_pos]
            masked_input_ids[idx, label_y_pos] = 103
            masked_input_ids[idx, label_gen_pos] = 103
        return masked_input_ids, mlm_labels

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None):
        masked_input_ids, mlm_labels = self._apply_masking_vectorized(input_ids, attention_mask)
        out = self.bert(input_ids=masked_input_ids, attention_mask=attention_mask, position_ids=position_ids)
        loss = nn.CrossEntropyLoss()(out.logits.view(-1, out.logits.size(-1)), mlm_labels.view(-1))
        if self.training:
            return type("ModelOutput", (), {"loss": loss, "raw_logits": None, "probs": None})()
        masked_input_ids = input_ids.clone()
        batch_size = input_ids.size(0)
        last_real_positions = attention_mask.sum(dim=1) - 1
        label_positions = last_real_positions - 1
        masked_input_ids[range(batch_size), label_positions] = 103
        out = self.bert(input_ids=masked_input_ids, attention_mask=attention_mask, position_ids=position_ids)
        vocab_logits = out.logits[range(batch_size), label_positions]
        raw_logits = vocab_logits[:, self.label_token_ids]
        return type("ModelOutput", (), {"loss": loss, "raw_logits": raw_logits, "probs": torch.softmax(raw_logits, dim=-1)})()

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class GPTModule(_MetricMixin, pl.LightningModule):
    def __init__(
        self,
        model_size: str,
        model_type: str,
        labels: List[int],
        learning_rate: float = 2e-5,
        num_warmup_steps: int = 0,
        label_token_ids: Optional[List[int]] = None,
        vocab_size: int = GPT2_VOCAB_SIZE,
        position_id_policy: str = "default",
        pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_GPT2_MODEL,
        pretrained_model_revision: Optional[str] = DEFAULT_PRETRAINED_GPT2_REVISION,
        initialize_from_pretrained: bool = True,
        pretrained_config: Optional[Dict[str, Any]] = None,
        ignore_pad_boundary_loss: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.position_id_policy = position_id_policy  # "default" (arange) | "mask_derived"
        self.model_size = model_size
        self.model_type = model_type
        self.labels = labels
        self.num_labels = len(labels)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.label_token_ids = label_token_ids or []
        self.ignore_pad_boundary_loss = ignore_pad_boundary_loss
        self._reset_epoch_cache()
        if model_size == "pretrained" and initialize_from_pretrained:
            if not _is_immutable_hf_revision(pretrained_model_revision):
                raise ValueError(
                    "pretrained GPT-2 requires a 40-character lowercase "
                    "commit SHA"
                )
            self.gpt2 = GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path,
                revision=pretrained_model_revision,
            )
        elif model_size == "pretrained":
            if not pretrained_config:
                raise ValueError(
                    "checkpoint reconstruction requires pretrained_config"
                )
            self.gpt2 = GPT2LMHeadModel(
                GPT2Config.from_dict(pretrained_config)
            )
        else:
            config = GPT2Config(vocab_size=vocab_size, n_positions=MAX_POSITION_EMBEDDINGS, **MODEL_CONFIGS["gpt2"][model_size])
            self.gpt2 = GPT2LMHeadModel(config)
        if vocab_size > self.gpt2.config.vocab_size:
            self.gpt2.resize_token_embeddings(vocab_size)

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None):
        if position_ids is None and getattr(self, "position_id_policy", "default") == "mask_derived":
            # consistency-trained checkpoints: mask-derived positions are the
            # module-internal default at BOTH train and inference time. An
            # explicitly passed position_ids (e.g. the diagnostic's "arange"
            # arm) always overrides.
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        lm_labels = input_ids.clone()
        lm_labels[attention_mask == 0] = -100
        if self.ignore_pad_boundary_loss:
            first_real = attention_mask.long().argmax(dim=1)
            lm_labels[
                torch.arange(input_ids.size(0), device=input_ids.device),
                first_real,
            ] = -100
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels, position_ids=position_ids)
        if self.training:
            return type("ModelOutput", (), {"loss": outputs.loss, "raw_logits": None, "probs": None})()
        if self.model_type == "pseudo-gen-GPT2":
            out2 = self.gpt2(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1],
                             position_ids=position_ids[:, :-1] if position_ids is not None else None)
            last_logits = out2.logits[:, -1, :]
            raw_logits = last_logits[:, self.label_token_ids]
            return type("ModelOutput", (), {"loss": outputs.loss, "raw_logits": raw_logits, "probs": torch.softmax(raw_logits, dim=-1)})()
        batch_size, T = input_ids.size()
        # Process each label separately to avoid OOM from expanding batch × num_labels
        all_label_scores = []
        for label_idx, label_token_id in enumerate(self.label_token_ids):
            modified_ids = input_ids.clone()
            first_real_positions = attention_mask.int().argmax(dim=1)
            second_real_positions = first_real_positions + 1
            modified_ids[torch.arange(batch_size, device=input_ids.device), second_real_positions] = label_token_id
            out = self.gpt2(input_ids=modified_ids, attention_mask=attention_mask, position_ids=position_ids)
            logits_all = out.logits[:, :-1, :]
            targets = modified_ids[:, 1:]
            mask = attention_mask[:, 1:].float()
            log_probs = torch.log_softmax(logits_all, dim=-1)
            tok_lp = torch.gather(log_probs, 2, targets.unsqueeze(2)).squeeze(2)
            score = (tok_lp * mask).sum(dim=1)
            all_label_scores.append(score)
        raw_logits = torch.stack(all_label_scores, dim=1)
        return type("ModelOutput", (), {"loss": outputs.loss, "raw_logits": raw_logits, "probs": torch.softmax(raw_logits, dim=-1)})()

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def setup_tokenizer(
    model_type: str,
    labels: List[int],
    pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_GPT2_MODEL,
    pretrained_model_revision: Optional[str] = None,
):
    if model_type in ["pseudo-gen-MLM", "BERT"]:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer_kwargs = {"padding_side": "left"}
        if pretrained_model_revision:
            tokenizer_kwargs["revision"] = pretrained_model_revision
        tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path,
            **tokenizer_kwargs,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    label_token_ids = None
    if model_type in ["pseudo-gen-MLM", "pure-gen-GPT2", "pseudo-gen-GPT2"]:
        tokens = ["[label_gen]"] + [f"[label_{label}]" for label in labels]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        label_token_ids = [tokenizer.convert_tokens_to_ids(f"[label_{label}]") for label in labels]
    return tokenizer, label_token_ids


def load_data(data_path: str, num_samples: Optional[int] = None, use_test_as_val: bool = False):
    tt, tl, vt, vl = load_train_val_data(
        data_path,
        use_test_as_val=use_test_as_val,
        val_split_size=VAL_SPLIT_SIZE,
        random_state=RANDOM_STATE,
    )
    if num_samples and num_samples > 0 and num_samples < len(tt):
        idxs = random.sample(range(len(tt)), num_samples)
        tt = [tt[i] for i in idxs]
        tl = [tl[i] for i in idxs]
    return tt, tl, vt, vl


def build_model(
    model_type: str,
    model_size: str,
    labels: List[int],
    learning_rate: float,
    num_warmup_steps: int,
    label_token_ids: Optional[List[int]],
    vocab_size: int,
    position_id_policy: str = "default",
    pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_GPT2_MODEL,
    pretrained_model_revision: Optional[str] = DEFAULT_PRETRAINED_GPT2_REVISION,
    ignore_pad_boundary_loss: bool = False,
):
    if model_type == "BERT":
        return BertModule(model_size, labels, learning_rate, num_warmup_steps)
    if model_type == "pseudo-gen-MLM":
        return MLMModule(model_size, model_type, labels, learning_rate, num_warmup_steps, label_token_ids, vocab_size)
    return GPTModule(
        model_size,
        model_type,
        labels,
        learning_rate,
        num_warmup_steps,
        label_token_ids,
        vocab_size,
        position_id_policy=position_id_policy,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        pretrained_model_revision=pretrained_model_revision,
        ignore_pad_boundary_loss=ignore_pad_boundary_loss,
    )


def train_model(args) -> None:
    run_dir = Path(args.output_root) / "train" / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    train_texts, train_labels, val_texts, val_labels = load_data(
        args.data_path,
        args.num_samples,
        args.use_test_as_val,
    )
    labels = sorted({int(label) for label in train_labels})
    is_pretrained_gpt2 = (
        args.model_size == "pretrained"
        and args.model_type in ("pure-gen-GPT2", "pseudo-gen-GPT2")
    )
    is_gpt2_family = args.model_type in (
        "pure-gen-GPT2", "pseudo-gen-GPT2"
    )
    pretrained_revision = (
        args.pretrained_model_revision if is_gpt2_family else None
    )
    tokenizer_source = (
        args.pretrained_model_name_or_path
        if is_gpt2_family
        else "bert-base-uncased"
    )
    tokenizer, label_token_ids = setup_tokenizer(
        args.model_type,
        labels,
        pretrained_model_name_or_path=tokenizer_source,
        pretrained_model_revision=pretrained_revision,
    )
    tokenizer.save_pretrained(run_dir / "tokenizer")
    label_mapping_path = run_dir / "label_mapping.json"
    write_json_atomic(
        {"labels": labels, "label_token_ids": label_token_ids},
        label_mapping_path,
    )

    data_module = TextDataModule((train_texts, train_labels), (val_texts, val_labels), tokenizer, args.model_type, args.batch_size)
    model = build_model(
        args.model_type,
        args.model_size,
        labels,
        args.learning_rate,
        args.num_warmup_steps,
        label_token_ids,
        len(tokenizer),
        position_id_policy=getattr(
            args, "position_id_policy", "default"
        ),
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_model_revision=args.pretrained_model_revision,
        ignore_pad_boundary_loss=args.ignore_pad_boundary_loss,
    )

    train_config = vars(args).copy()
    train_config.update(
        {
            "labels": labels,
            "label_token_ids": label_token_ids,
            "vocab_size": len(tokenizer),
            "dataset_source": get_dataset_source(args.data_path),
            "train_data_sha256": sha256_json(
                {"texts": train_texts, "labels": train_labels}
            ),
            "validation_data_sha256": sha256_json(
                {"texts": val_texts, "labels": val_labels}
            ),
            "resolved_training_position_id_policy": (
                "arange"
                if args.position_id_policy == "default"
                else args.position_id_policy
            ),
            "tokenizer_artifact_sha256": sha256_tree(
                run_dir / "tokenizer"
            ),
            "label_mapping_artifact_sha256": sha256_file(
                label_mapping_path
            ),
        }
    )
    if is_pretrained_gpt2:
        resolved_revision = (
            getattr(model.gpt2.config, "_commit_hash", None)
            or args.pretrained_model_revision
        )
        if resolved_revision != args.pretrained_model_revision:
            raise ValueError(
                "resolved pretrained GPT-2 revision differs from the "
                f"requested revision: {resolved_revision!r} != "
                f"{args.pretrained_model_revision!r}"
            )
        train_config.update(
            {
                "pretrained_model_name_or_path": (
                    args.pretrained_model_name_or_path
                ),
                "pretrained_model_revision": (
                    args.pretrained_model_revision
                ),
                "resolved_pretrained_model_revision": (
                    resolved_revision
                ),
                "pretrained_config": model.gpt2.config.to_dict(),
                "preprocessing_contract": {
                    "version": "pretrained_gpt2_b2_v1",
                    "padding_side": "left",
                    "pad_token_id": tokenizer.pad_token_id,
                    "pad_token": tokenizer.pad_token,
                    "sequence_length": MAX_POSITION_EMBEDDINGS,
                    "text_truncation_length": (
                        MAX_POSITION_EMBEDDINGS - 2
                    ),
                    "classification_interface": (
                        "pseudo_gen_next_label_token_v1"
                    ),
                    "ignore_pad_boundary_loss": bool(
                        args.ignore_pad_boundary_loss
                    ),
                },
            }
        )
    write_json_atomic(train_config, run_dir / "train_config.json")
    write_json_atomic(environment_fingerprint(), run_dir / "environment.json")
    write_json_atomic({"status": "RUNNING", "time": __import__("time").time()}, run_dir / "status.json")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{args.run_name}" + "_{val_acc:.3f}_{val_macro_f1:.3f}_{val_weighted_f1:.3f}_{val_auroc:.3f}_{epoch:02d}",
        monitor="val_weighted_f1",
        mode="max",
        save_top_k=CHECKPOINT_TOP_K,
        save_last=True,
    )
    callbacks = [checkpoint_callback]
    if args.early_stopping_patience and args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_weighted_f1",
            mode="max",
            patience=args.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    logger = TensorBoardLogger(str(log_dir), name=args.run_name)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        precision=TRAINER_PRECISION,
        accelerator="auto",
        # Single-GPU per job by design. Parallelism comes from running many
        # independent checkpoints across devices, not from distributed data parallel training
        # within one training job. devices=1 also prevents a direct (non-pool)
        # invocation on a multi-GPU box from silently launching DDP across all
        # visible GPUs (which would inject all-reduce nondeterminism + change
        # effective batch size). Under the pool, CUDA_VISIBLE_DEVICES already
        # pins one GPU; devices=1 makes the single-device intent explicit.
        devices=1,
        strategy=args.strategy,
        log_every_n_steps=50,
    )
    ckpt_path = None
    if args.resume == "auto" and (ckpt_dir / "last.ckpt").exists():
        ckpt_path = str(ckpt_dir / "last.ckpt")
    elif args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not Path(best_model_path).is_file():
        raise RuntimeError(
            "training completed without a concrete best checkpoint: "
            f"{best_model_path!r}"
        )
    best_model_path = str(Path(best_model_path).resolve())
    best_checkpoint_sha256 = sha256_file(best_model_path)
    write_json_atomic(
        {
            "best_model_path": best_model_path,
            "best_model_score": float(
                checkpoint_callback.best_model_score or 0.0
            ),
            "best_checkpoint_sha256": best_checkpoint_sha256,
        },
        run_dir / "best_checkpoint.json",
    )
    write_json_atomic({"status": "SUCCEEDED", "time": __import__("time").time()}, run_dir / "status.json")
    mark_success(
        run_dir,
        {
            "best_model_path": best_model_path,
            "best_checkpoint_sha256": best_checkpoint_sha256,
        },
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=MODEL_TYPES, required=True)
    p.add_argument("--model_size", choices=MODEL_SIZES, required=True)
    p.add_argument("--experiment_id", default=None)
    p.add_argument("--training_regime", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--num_warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--seed_profile", choices=["strict", "production_like", "none"], default="strict")
    p.add_argument("--position_id_policy", default="default", choices=["default", "mask_derived"])
    p.add_argument(
        "--pretrained_model_name_or_path",
        default=DEFAULT_PRETRAINED_GPT2_MODEL,
    )
    p.add_argument(
        "--pretrained_model_revision",
        default=DEFAULT_PRETRAINED_GPT2_REVISION,
        help="immutable Hugging Face commit required for model_size=pretrained",
    )
    p.add_argument(
        "--ignore_pad_boundary_loss",
        action="store_true",
        help=(
            "exclude the first real token from causal-LM loss when its "
            "predecessor is padding"
        ),
    )
    p.add_argument("--use_test_as_val", action="store_true")
    p.add_argument("--output_root", default="artifacts")
    p.add_argument("--run_id", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--resume", choices=["none", "auto"], default="auto")
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--strategy", default="auto")
    p.add_argument("--early_stopping_patience", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    apply_seed_profile(args.seed, args.seed_profile)
    train_model(args)


if __name__ == "__main__":
    main()
