import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallTextEncoder(nn.Module):
    """Small trainable text encoder - balanced with vision encoder"""
    def __init__(self, vocab_size, embed_dim=256, depth=4, num_heads=8, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add positional embeddings
        embeddings = embeddings + self.positional_embedding[:, :seq_len, :]
        
        # Create padding mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        
        # Encode
        encoded = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        return self.layer_norm(encoded)


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
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        patches = self.patch_embed(pixel_values)
        bsz, dim, h, w = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        if h * w != self.num_patches:
            raise ValueError("Input resolution does not match initialized positional embeddings.")
        patches = patches + self.positional_embedding
        encoded = self.encoder(patches)
        return self.layer_norm(encoded)


class DiagramExplainerModel(nn.Module):
    def __init__(
        self,
        vocab_size=256000,
        text_embed_dim=256,
        text_depth=4,
        text_heads=8,
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
        self.text_encoder = SmallTextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            depth=text_depth,
            num_heads=text_heads,
        )
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
        
        # Cross-modal attention (text attends to image)
        self.cross_attention = nn.MultiheadAttention(fusion_dim, fusion_heads, dropout=0.0, batch_first=True)
        self.cross_attention_norm = nn.LayerNorm(fusion_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=decoder_heads,
            dim_feedforward=int(fusion_dim * decoder_mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Decoder uses same embeddings as text encoder
        self.decoder_input_projection = nn.Linear(text_dim, fusion_dim)
        self.decoder_output_projection = nn.Linear(fusion_dim, text_dim)

        # Output head shares weights with text encoder embeddings
        self.output_head = nn.Linear(text_dim, vocab_size, bias=False)
        self.output_head.weight = self.text_encoder.token_embedding.weight
        
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

        # Cross-modal attention: text attends to image
        attn_output, _ = self.cross_attention(
            query=text_embeddings,
            key=image_features,
            value=image_features,
            need_weights=False,
        )
        fused_embeddings = self.cross_attention_norm(text_embeddings + attn_output)

        decoder_embeddings = self.text_encoder.token_embedding(decoder_input_ids)
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

