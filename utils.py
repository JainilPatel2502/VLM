from pathlib import Path
import torch
import sentencepiece as spm


class CustomTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_token_id = self.sp.pad_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        self.pad_token = "<pad>"
    
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = [self.sp.encode_as_ids(text) for text in texts]
        
        if truncation:
            encoded = [ids[:max_length] for ids in encoded]
        
        if padding:
            max_len = max(len(ids) for ids in encoded)
            if truncation:
                max_len = min(max_len, max_length)
            encoded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded]
        
        attention_mask = [[1 if token_id != self.pad_token_id else 0 for token_id in ids] 
                          for ids in encoded]
        
        result = {"input_ids": encoded, "attention_mask": attention_mask}
        
        if return_tensors == "pt":
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        
        return result
    
    def decode(self, token_ids, skip_special_tokens):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if tid not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}]
        
        return self.sp.decode_ids(token_ids)
    
    def vocab_size(self):
        return self.sp.vocab_size()


def build_tokenizer():
    return CustomTokenizer("tokenizer.model")


def prepare_inputs(tokenizer, chart_infos, captions, max_encoder_tokens, max_decoder_tokens, device):
    encoder_texts = [info if (info and info.strip()) else caption for info, caption in zip(chart_infos, captions)]

    encoder_batch = tokenizer(encoder_texts, padding=True, truncation=True, max_length=max_encoder_tokens, return_tensors="pt")
    decoder_batch = tokenizer(captions, padding=True, truncation=True, max_length=max_decoder_tokens, return_tensors="pt")

    input_ids = decoder_batch["input_ids"]
    attention_mask = decoder_batch["attention_mask"]

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


def save_checkpoint(output_dir, step, samples_seen, model, optimizer):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    torch.save(
        {
            "global_step": step,
            "samples_seen": samples_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def maybe_synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def find_latest_checkpoint(directory):
    if not directory.exists():
        return None
    checkpoint_path = directory / "checkpoint_latest.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    return None


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
