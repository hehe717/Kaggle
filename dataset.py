import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import ToTensor, transforms
import torch.nn.functional as F
import pandas as pd

class PlantDataset(Dataset):
    def __init__(self, dataframe, img_dir, input_size, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir

        default_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor()
        ])

        self.transform = transform if transform else default_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_id'] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
        label = self.dataframe.iloc[idx][cols].values.astype(float)

        return image, torch.tensor(label)


def calculate_embeddings_and_remove_similar(dataset, cosine_threshold=0.9):
    print("Filtering..")

    embeddings = []
    labels = []

    # DataLoader를 사용하여 임베딩 계산
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image, label in dataloader:
        embeddings.append(image.view(-1))  # Flatten to 1D
        labels.append(label.view(-1))

    embeddings = torch.stack(embeddings)  # (num_samples, flattened_dim)
    labels = torch.stack(labels)  # (num_samples, num_classes)

    # 제거할 인덱스 저장
    remove_indices = set()

    # 코사인 유사도를 계산하여 유사도가 높은데 라벨이 다른 경우 찾기
    num_samples = embeddings.size(0)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            cosine_sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            if cosine_sim >= cosine_threshold and not torch.equal(labels[i], labels[j]):
                remove_indices.add(i)
                remove_indices.add(j)

    # 데이터프레임에서 해당 인덱스 제거
    keep_indices = [i for i in range(num_samples) if i not in remove_indices]
    filtered_dataframe = dataset.dataframe.iloc[keep_indices].reset_index(drop=True)

    print("Filtering done.")
    return PlantDataset(filtered_dataframe, dataset.img_dir, input_size=(224, 224), transform=dataset.transform)
