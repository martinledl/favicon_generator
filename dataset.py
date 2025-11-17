import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from io import BytesIO


class LogoDataset(Dataset):
    def __init__(self, df, adjective_to_idx, industry_to_idx, img_size=128):
        self.images = df["image"].tolist()
        self.adjectives = df["adjective"].tolist()
        self.industries = df["industry"].tolist()
        self.adjective_to_idx = adjective_to_idx
        self.industry_to_idx = industry_to_idx

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        row = {
            "image": self.images[idx],
            "adjective": self.adjectives[idx],
            "industry": self.industries[idx]
        }

        # load image
        img = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
        img = self.transform(img)

        # map strings → integers
        adjective_id = self.adjective_to_idx[row["adjective"]]
        industry_id = self.industry_to_idx[row["industry"]]

        # return image + conditioning
        return img, torch.tensor(adjective_id, dtype=torch.long), torch.tensor(industry_id, dtype=torch.long)