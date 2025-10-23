from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer


def build_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_inputs(
    tokenizer: AutoTokenizer,
    chart_infos: List[str],
    captions: List[str],
    max_encoder_tokens: int,
    max_decoder_tokens: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    encoder_texts = [info if (info and info.strip()) else caption for info, caption in zip(chart_infos, captions)]

    encoder_batch = tokenizer(
        encoder_texts,
        padding=True,
        truncation=True,
        max_length=max_encoder_tokens,
        return_tensors="pt",
    )

    decoder_batch = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_decoder_tokens,
        return_tensors="pt",
    )

    input_ids = decoder_batch["input_ids"]
    attention_mask = decoder_batch["attention_mask"]

    if input_ids.size(1) < 2:
        raise ValueError(
            "Decoder sequence length must be at least 2 tokens (including BOS/EOS). Increase max_decoder_tokens or ensure captions include content."
        )

    decoder_input_ids = input_ids[:, :-1]
    decoder_attention_mask = attention_mask[:, :-1]

    labels = input_ids[:, 1:].clone()
    labels_attention_mask = attention_mask[:, 1:]
    labels[labels_attention_mask == 0] = -100

    encoder_batch = {key: value.to(device) for key, value in encoder_batch.items()}
    decoder_inputs = {
        "decoder_input_ids": decoder_input_ids.to(device),
        "decoder_attention_mask": decoder_attention_mask.to(device),
    }
    labels = labels.to(device)
    return encoder_batch, decoder_inputs, labels


def save_checkpoint(
    output_dir: Path,
    step: int,
    samples_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    torch.save(
        {
            "global_step": step,
            "samples_seen": samples_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": model_name,
        },
        checkpoint_path,
    )
    return checkpoint_path


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def find_latest_checkpoint(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    checkpoint_path = directory / "checkpoint_latest.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    return None


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
