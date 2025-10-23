from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import DiagramExplainerModel
from utils import build_tokenizer

# Configuration
CHECKPOINT_PATH = "checkpoints/checkpoint_latest.pt"
IMAGE_PATH = "image.jpg"
INPUT_TEXT = """* Title: pGMM Kernel Regression
*Type*: Line chart
*Legends*: Not specified
*Labels*: "p = 80," "p = 60," "p = 40," and "p = 20" annotate the individual data series.
*Data Comparison*: The data series exhibit different MSE (Mean Squared Error) values, where higher values of \(p\) result in larger MSE deviations at higher \(\lambda\) values (e.g., near \(10^{0}\)), while their values are nearly parallel in the lower \(\lambda\) range.
*Data Correlations/Trends*: All data series show a U-shaped pattern, indicating that MSE decreases initially as \(\lambda\) increases, reaches a minimum, and then increases again. This occurs consistently across all values of \(p\), with the magnitude of deviation varying.
Axes:
- X-axis (λ): Logarithmic scale from 10^-5 to 10^0
- Y-axis (MSE): Linear scale from 0.011 to 0.016
- Title: "pGMM Kernel Regression"
- Subtitle: "Mimage"

Retrieve Value:
Initial points (λ = 10^-5):
- p=80: ~0.0136
- p=20: ~0.0131
- p=60: ~0.0129
- p=40: ~0.0125

Middle points (λ = 10^-2):
- p=80: ~0.0137
- p=20: ~0.0132
- p=60: ~0.0130
- p=40: ~0.0126

End points (λ = 10^0):
All series converge to ~0.016

Find Extremum:
Minimum:
- p=80: ~0.0136 at λ=10^-5
- p=20: ~0.0131 at λ=10^-5
- p=60: ~0.0129 at λ=10^-5
- p=40: ~0.0125 at λ=10^-5
"""
MAX_LENGTH = 256
TEMPERATURE = 0.5
IMAGE_SIZE = 224


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint: step={checkpoint.get('global_step', 0)}")
    return checkpoint


def prepare_image(image_path, image_size=(224, 224)):
    """Load and preprocess image."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def prepare_text(text, tokenizer, max_length=256, device="cuda"):
    """Tokenize input text."""
    encoded = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def generate_caption(model, tokenizer, image_tensor, encoder_text, max_length=128, temperature=0.8, device="cuda"):
    """Generate caption using the model."""
    model.eval()
    
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        encoder_batch = prepare_text(encoder_text, tokenizer, max_length=256, device=device)
        
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.pad_token_id
        decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        
        generated_tokens = []
        
        for _ in range(max_length):
            outputs = model(
                pixel_values=image_tensor.to(device),
                decoder_input_ids=decoder_input_ids,
                encoder_input_ids=encoder_batch["input_ids"],
                encoder_attention_mask=encoder_batch.get("attention_mask"),
            )
            
            logits = outputs["logits"][:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token.item())
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
tokenizer = build_tokenizer()

model = DiagramExplainerModel(
    vocab_size=8192,  # Custom tokenizer vocabulary size
    text_embed_dim=256,
    text_depth=8,  # Deeper text encoder
    text_heads=8,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    patch_size=16,
    vision_embed_dim=256,
    vision_depth=8,  # Deeper vision encoder
    vision_heads=8,
    fusion_dim=256,
    fusion_heads=4,
    decoder_layers=4,  # Deeper decoder
    decoder_heads=4,
    decoder_mlp_ratio=3.0,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
print("Model loaded!\n")

print(f"Image: {IMAGE_PATH}")
print(f"Text: {INPUT_TEXT[:80]}...")
print("\nGenerating caption...\n")

image_tensor = prepare_image(IMAGE_PATH, image_size=(IMAGE_SIZE, IMAGE_SIZE))
caption = generate_caption(model, tokenizer, image_tensor, INPUT_TEXT, MAX_LENGTH, TEMPERATURE, device)

print(f"Generated Caption:\n>>> {caption}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()




