import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Dataloader import StreamingChartCapDataset, collate_fn, image_transforms
from model import DiagramExplainerModel
from utils import (
    build_tokenizer,
    count_model_parameters,
    find_latest_checkpoint,
    maybe_synchronize,
    prepare_inputs,
    save_checkpoint,
)

torch.set_float32_matmul_precision("medium")

MODEL_NAME = "google/gemma-3-270m"
BATCH_SIZE = 2
MAX_STEPS = 5000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8
GRAD_CLIP_NORM = 1.0
IMAGE_SIZE = (224, 224)
PATCH_SIZE = 16
VISION_EMBED_DIM = 256
VISION_DEPTH = 6
VISION_HEADS = 8
FUSION_DIM = 384
FUSION_HEADS = 8
DECODER_LAYERS = 4
DECODER_HEADS = 6
DECODER_MLP_RATIO = 4.0
MAX_ENCODER_TOKENS = 512
MAX_DECODER_TOKENS = 512
START_INDEX = 0
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 100
RESUME_FROM = None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(CHECKPOINT_DIR)
    resume_state = None
    model_name = MODEL_NAME
    resume_path = Path(RESUME_FROM) if RESUME_FROM else find_latest_checkpoint(checkpoint_dir)
    if resume_path:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu")
        checkpoint_model_name = resume_state.get("model_name", MODEL_NAME)
        if checkpoint_model_name != MODEL_NAME:
            print(
                f"Notice: overriding MODEL_NAME ({MODEL_NAME}) with checkpoint model '{checkpoint_model_name}' for compatibility."
            )
            model_name = checkpoint_model_name

    tokenizer = build_tokenizer(model_name)

    model = DiagramExplainerModel(
        gemma_model_name=model_name,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        vision_embed_dim=VISION_EMBED_DIM,
        vision_depth=VISION_DEPTH,
        vision_heads=VISION_HEADS,
        fusion_dim=FUSION_DIM,
        fusion_heads=FUSION_HEADS,
        decoder_layers=DECODER_LAYERS,
        decoder_heads=DECODER_HEADS,
        decoder_mlp_ratio=DECODER_MLP_RATIO,
    ).to(device)
    trainable_params, frozen_params = count_model_parameters(model)
    total_params = trainable_params + frozen_params

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        global_step = int(resume_state["global_step"])
        samples_seen = int(resume_state["samples_seen"])

        if device.type == "cuda":
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
    else:
        global_step = 0
        samples_seen = START_INDEX

    dataset = StreamingChartCapDataset(split="train", transform=image_transforms, start_idx=samples_seen)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    )

    # Setup mixed precision training
    scaler = torch.amp.GradScaler("cuda")
    
    model.train()

    print(
        f"Starting training at step {global_step}, samples_seen={samples_seen}, device={device}, tokenizer_pad={tokenizer.pad_token}"
    )
    print(
        f"Model parameters: total={total_params:,} | trainable={trainable_params:,} | frozen={frozen_params:,}"
    )

    skipped_batches = 0
    
    try:
        for batch in dataloader:
            if global_step >= MAX_STEPS:
                break

            try:
                images = batch["image"].to(device)
                chart_infos = batch["chartInfo"]
                captions = batch["caption"]

                encoder_batch, decoder_inputs, labels = prepare_inputs(
                    tokenizer,
                    chart_infos,
                    captions,
                    MAX_ENCODER_TOKENS,
                    MAX_DECODER_TOKENS,
                    device,
                )

                optimizer.zero_grad(set_to_none=True)

                maybe_synchronize(device)
                forward_start = time.perf_counter()

                # Forward pass with mixed precision
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        pixel_values=images,
                        decoder_input_ids=decoder_inputs["decoder_input_ids"],
                        encoder_input_ids=encoder_batch["input_ids"],
                        encoder_attention_mask=encoder_batch.get("attention_mask"),
                        decoder_attention_mask=decoder_inputs["decoder_attention_mask"],
                        labels=labels,
                    )
                    loss = outputs["loss"]

                maybe_synchronize(device)
                forward_end = time.perf_counter()

                backward_start = forward_end
                
                # Backward pass with gradient scaling and clipping
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

                maybe_synchronize(device)
                backward_end = time.perf_counter()

                step_ms = (backward_end - forward_start) * 1000.0
                forward_ms = (forward_end - forward_start) * 1000.0
                backward_ms = (backward_end - backward_start) * 1000.0

                global_step += 1
                samples_seen += len(captions)

                print(
                    f"Step {global_step:06d} | Loss {loss.item():.4f} | Grad Norm {grad_norm:.4f} | Step {step_ms:.2f} ms | Forward {forward_ms:.2f} ms | Backward {backward_ms:.2f} ms"
                )

                if global_step % CHECKPOINT_EVERY == 0:
                    checkpoint_path = save_checkpoint(
                        checkpoint_dir,
                        global_step,
                        samples_seen,
                        model,
                        optimizer,
                        model_name,
                    )
                    print(f"Saved checkpoint to {checkpoint_path}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    print(f"\n⚠️  CUDA OOM Error at step {global_step}! Skipping batch and clearing cache...")
                    skipped_batches += 1
                    samples_seen += len(captions) if 'captions' in locals() else 0
                    
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Reset gradients
                    optimizer.zero_grad(set_to_none=True)
                    
                    print(f"Cleared cache. Total skipped batches: {skipped_batches}. Continuing...\n")
                    continue
                else:
                    # Re-raise if it's not a memory error
                    raise e

        else:
            print("DataLoader exhausted before reaching max_steps.")
            print(f"Total batches skipped due to OOM: {skipped_batches}")

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        checkpoint_path = save_checkpoint(
            checkpoint_dir,
            global_step,
            samples_seen,
            model,
            optimizer,
            model_name,
        )
        print(f"Saved checkpoint to {checkpoint_path}")
    finally:
        print("Training finished.")


if __name__ == "__main__":
    main()
