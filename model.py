import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def build_2d_sincos_position_embedding(
    grid_size,
    embed_dim,
    device=None,
):
    """Generate a 2D sin-cos positional embedding."""
    grid_h, grid_w = grid_size
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, grid_h, dtype=torch.float32, device=device),
        torch.linspace(0, 1, grid_w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))

    out_y = torch.einsum("hw,d->hwd", grid_y, omega)
    out_x = torch.einsum("hw,d->hwd", grid_x, omega)

    pos_y = torch.cat([out_y.sin(), out_y.cos()], dim=-1)
    pos_x = torch.cat([out_x.sin(), out_x.cos()], dim=-1)
    pos = torch.cat([pos_y, pos_x], dim=-1)
    pos = pos.view(grid_h * grid_w, embed_dim)
    return pos


class FrozenGemmaTextEncoder(nn.Module):
    def __init__(self, model_name="google/gemma-2b", trust_remote_code=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=False,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        embeddings = self.model.get_input_embeddings()
        self.embedding_dim = embeddings.embedding_dim
        self.vocab_size = embeddings.num_embeddings
        self.register_buffer("_embedding_weight", embeddings.weight.detach().clone())

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def get_embedding_weight(self):
        return self._embedding_weight


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size=(224, 224),
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        h, w = image_size
        assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.grid_size = (h // patch_size, w // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.positional_embedding = nn.Parameter(
            build_2d_sincos_position_embedding(self.grid_size, embed_dim),
            requires_grad=False,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        patches = self.patch_embed(pixel_values)
        bsz, dim, h, w = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        if h * w != self.num_patches:
            raise ValueError("Input resolution does not match initialized positional embeddings.")
        patches = patches + self.positional_embedding.unsqueeze(0)
        encoded = self.encoder(patches)
        return self.layer_norm(encoded)


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries,
        keys,
        values,
        key_padding_mask=None,
    ):
        attn_output, _ = self.attn(
            queries,
            keys,
            values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_output
        return self.norm(queries)


class DiagramExplainerModel(nn.Module):
    def __init__(
        self,
        gemma_model_name="google/gemma-3-270m",
        trust_remote_code=False,
        image_size=(224, 224),
        patch_size=16,
        vision_embed_dim=256,
        vision_depth=6,
        vision_heads=8,
        fusion_dim=512,
        fusion_heads=8,
        decoder_layers=6,
        decoder_heads=8,
        decoder_mlp_ratio=4.0,
    ):
        super().__init__()
        self.text_encoder = FrozenGemmaTextEncoder(gemma_model_name, trust_remote_code=trust_remote_code)
        text_dim = self.text_encoder.embedding_dim
        vocab_size = self.text_encoder.vocab_size

        self.vision_encoder = VisionTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_heads,
        )
        self.image_projection = nn.Linear(vision_embed_dim, fusion_dim)

        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.cross_attention = CrossModalAttentionBlock(fusion_dim, fusion_heads)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=decoder_heads,
            dim_feedforward=int(fusion_dim * decoder_mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        token_embedding = self.text_encoder.get_embedding_weight()
        self.token_embedding = nn.Embedding.from_pretrained(token_embedding, freeze=False)
        self.decoder_input_projection = nn.Linear(self.token_embedding.embedding_dim, fusion_dim)
        self.decoder_output_projection = nn.Linear(fusion_dim, self.token_embedding.embedding_dim)

        self.output_head = nn.Linear(self.token_embedding.embedding_dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values,
        decoder_input_ids,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        image_features = self.vision_encoder(pixel_values)
        image_features = self.image_projection(image_features)

        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids must be provided for cross-attention queries.")

        text_embeddings = self.text_encoder(encoder_input_ids, attention_mask=encoder_attention_mask)
        text_embeddings = self.text_projection(text_embeddings)

        fused_embeddings = self.cross_attention(
            queries=text_embeddings,
            keys=image_features,
            values=image_features,
        )

        decoder_embeddings = self.token_embedding(decoder_input_ids)
        decoder_embeddings = self.decoder_input_projection(decoder_embeddings)

        tgt_mask = self._generate_square_subsequent_mask(decoder_embeddings.size(1), decoder_embeddings.device)
        tgt_key_padding_mask = self._attention_mask_to_padding(decoder_attention_mask)
        memory_key_padding_mask = self._attention_mask_to_padding(encoder_attention_mask)

        decoder_output = self.decoder(
            tgt=decoder_embeddings,
            memory=fused_embeddings,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        decoder_output = self.decoder_output_projection(decoder_output)
        logits = self.output_head(decoder_output)

        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output

    @staticmethod
    def _generate_square_subsequent_mask(size, device):
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    @staticmethod
    def _attention_mask_to_padding(attention_mask):
        if attention_mask is None:
            return None
        return attention_mask == 0

