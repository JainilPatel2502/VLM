import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
import torchvision.transforms.functional as F
class StreamingChartCapDataset(Dataset):
    def __init__(self, split="train", transform=None, start_idx=0):
        self.dataset = load_dataset("junyoung-00/ChartCap", split=split, streaming=True)
        self.dataset = self.dataset.skip(start_idx)
        self.dataset_iter = iter(self.dataset)
        self.transform = transform
        self.start_idx = start_idx


    def __len__(self):
        return 1000000000

    def __getitem__(self, idx):
        sample = next(self.dataset_iter)
        image = sample["image"]
        if self.transform:
            image = self.transform(image)

        chart_info = sample.get("chart_info", "")
        caption = sample.get("caption", "")

        return {
            "image": image,
            "chartInfo": chart_info,
            "caption": caption
        }

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = StreamingChartCapDataset(split="train", transform=image_transforms, start_idx=0)

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    chart_info = [b["chartInfo"] for b in batch]
    captions = [b["caption"] for b in batch]
    return {"image": images, "chartInfo": chart_info, "caption": captions}

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)





# # testing
# for i, batch in enumerate(dataloader):
#     print(batch["chartInfo"], batch["caption"])
#     print(batch['image'].shape)
#     if i >= 100:
#         break