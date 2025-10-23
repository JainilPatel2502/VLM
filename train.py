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
    prepare_inputs,
    save_checkpoint,
)
torch.set_float32_matmul_precision("medium")
MODEL_NAME = "custom_tokenizer"  # Custom trained tokenizer
VOCAB_SIZE = 8192  # Custom tokenizer vocabulary size
BATCH_SIZE = 2
MAX_STEPS = 10000  # 20k images with batch size 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05  # Stronger regularization for small dataset
BETAS = (0.9, 0.999)
EPS = 1e-8
IMAGE_SIZE = (224, 224)
PATCH_SIZE = 16
# Text encoder (SCALED UP 2.5x)
TEXT_EMBED_DIM = 384
TEXT_DEPTH = 10
TEXT_HEADS = 12
# Vision encoder (SCALED UP 2.5x)
VISION_EMBED_DIM = 384
VISION_DEPTH = 10
VISION_HEADS = 12
# Fusion and decoder (SCALED UP 2.5x)
FUSION_DIM = 384
FUSION_HEADS = 6
DECODER_LAYERS = 6
DECODER_HEADS = 6
DECODER_MLP_RATIO = 4.0
MAX_ENCODER_TOKENS = 512
MAX_DECODER_TOKENS = 512
START_INDEX = 0
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 100
VAL_EVERY = 20
RESUME_FROM = None



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = Path(CHECKPOINT_DIR)
resume_state = None
resume_path = Path(RESUME_FROM) if RESUME_FROM else find_latest_checkpoint(checkpoint_dir)
if resume_path:
    if not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    print(f"Resuming from checkpoint {resume_path}")
    resume_state = torch.load(resume_path, map_location="cpu")

tokenizer = build_tokenizer()

model = DiagramExplainerModel(
    vocab_size=VOCAB_SIZE,
    text_embed_dim=TEXT_EMBED_DIM,
    text_depth=TEXT_DEPTH,
    text_heads=TEXT_HEADS,
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
total_params = count_model_parameters(model)

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

val_dataset = StreamingChartCapDataset(split="train", transform=image_transforms, start_idx=45000)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
)

# Setup mixed precision training
scaler = torch.amp.GradScaler("cuda")

model.train()

print(
    f"Starting training at step {global_step}, samples_seen={samples_seen}, device={device}, tokenizer_pad={tokenizer.pad_token}"
)
print(f"Model parameters: {total_params:,}")

try:
    for batch in dataloader:
        if global_step >= MAX_STEPS:
            break

        step_start = time.time()

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

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        samples_seen += len(captions)

        step_time = (time.time() - step_start) * 1000
        print(f"Step {global_step:06d} | Loss {loss.item():.4f} | Time {step_time:.2f}ms")

        if global_step % VAL_EVERY == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    if val_steps >= 10:
                        break
                    val_images = val_batch["image"].to(device)
                    val_chart_infos = val_batch["chartInfo"]
                    val_captions = val_batch["caption"]
                    
                    val_encoder_batch, val_decoder_inputs, val_labels = prepare_inputs(
                        tokenizer,
                        val_chart_infos,
                        val_captions,
                        MAX_ENCODER_TOKENS,
                        MAX_DECODER_TOKENS,
                        device,
                    )
                    
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        val_outputs = model(
                            pixel_values=val_images,
                            decoder_input_ids=val_decoder_inputs["decoder_input_ids"],
                            encoder_input_ids=val_encoder_batch["input_ids"],
                            encoder_attention_mask=val_encoder_batch.get("attention_mask"),
                            decoder_attention_mask=val_decoder_inputs["decoder_attention_mask"],
                            labels=val_labels,
                        )
                        val_loss += val_outputs["loss"].item()
                    val_steps += 1
            
            val_loss = val_loss / val_steps
            print(f">>> Validation Loss: {val_loss:.4f}")
            model.train()

        if global_step % CHECKPOINT_EVERY == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_dir,
                global_step,
                samples_seen,
                model,
                optimizer,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training completed!")

except KeyboardInterrupt:
    print("Training interrupted by user. Saving checkpoint...")
    checkpoint_path = save_checkpoint(
        checkpoint_dir,
        global_step,
        samples_seen,
        model,
        optimizer,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
finally:
    print("Training finished.")


