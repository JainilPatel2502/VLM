from torch.utils.data import DataLoader
from Dataloader import StreamingChartCapDataset, collate_fn
from tqdm import tqdm


OUTPUT_FILE = "corpus.txt"
NUM_SAMPLES = 20000
BATCH_SIZE = 32


print(f"Preparing corpus from {NUM_SAMPLES} samples...")
    

dataset = StreamingChartCapDataset(split="train", transform=None, start_idx=0)
dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: x, 
    )
    
samples_processed = 0
    
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for batch in tqdm(dataloader, desc="Processing batches"):
        for sample in batch:
            if samples_processed >= NUM_SAMPLES:
                break
                
            chart_info = sample.get("chartInfo", "")
            caption = sample.get("caption", "")
            text = f"{chart_info}\n{caption}\n"
            f.write(text)
                
            samples_processed += 1
            
        if samples_processed >= NUM_SAMPLES:
            break
    
print(f"\nCorpus saved to {OUTPUT_FILE}")
print(f" Total samples: {samples_processed}")