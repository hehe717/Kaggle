import pandas as pd
import torch

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PlantDataset
from torchinfo import summary

from model.new_vit import VitClassifier
from model.vit import VisionTransformer

df_train = pd.read_csv('data/train.csv')
print(df_train.head())

img_size = 224
dataset = PlantDataset(df_train, 'data/images', input_size=(img_size, img_size), transform=None)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# img_size, patch_size, num_heads, intermediate_size, num_layers, hidden_size, num_classes
model = VitClassifier(
    img_size=img_size,
    patch_size=16,
    num_heads=12,
    intermediate_size=3072,
    num_layers=12,
    hidden_size=768,
    num_classes=4
)


# model = VisionTransformer(
#     img_size=img_size,
#     in_channels=3,
#     patch_size=16,
#     embed_dim=768,
#     num_heads=12,
#     mlp_ratio=3,
#     depth=12
# )

# model = ResNeXt50(num_classes=4)

# Training code
epochs = 100
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

if torch.cuda.is_available():
    device = torch.device('cuda')

model.to(device)
print("Device: ", device)
print(summary(model))

writer = SummaryWriter()

import torch.profiler

min_loss = 99999
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("Loss: ", loss.item())

    # Update weights
    # avg loss as float

    avg_loss = running_loss / len(data_loader)
    scheduler.step(avg_loss)
    print(f"Epoch: {epoch}, Loss: {avg_loss}")

    # Save model for best
    if avg_loss < min_loss:
        min_loss = avg_loss
        print("Saved best_model.pth")
        torch.save(model.state_dict(), 'best_model.pth')

writer.close()