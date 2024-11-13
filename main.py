import pandas as pd
import torch

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PlantDataset
from torchinfo import summary
from model.vit import VisionTransformer

df_train = pd.read_csv('data/train.csv')
print(df_train.head())

img_size = 224
dataset = PlantDataset(df_train, 'data/images', input_size=(img_size, img_size), transform=None)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = VisionTransformer(
    img_size=img_size,
    in_channels=3,
    patch_size=32,
    embed_dim=1024,
    num_heads=32,
    mlp_ratio=3,
    depth=16
)

# model = CCT(
#     img_size = (224, 224),
#     embedding_dim = 384,
#     n_conv_layers = 2,
#     kernel_size = 7,
#     stride = 2,
#     padding = 3,
#     pooling_kernel_size = 3,
#     pooling_stride = 2,
#     pooling_padding = 1,
#     num_layers = 14,
#     num_heads = 6,
#     mlp_ratio = 3.,
#     num_classes = 4,
#     positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
# )
# model = se_resnext50_32x4d()
# model = ResNeXt50(num_classes=4)

# Training code
epochs = 100
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

if torch.cuda.is_available():
    device = torch.device('cuda')

model.to(device)
print("Device: ", device)
print(summary(model))

writer = SummaryWriter()

import torch.profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA  # GPU 프로파일링 (CUDA 사용 시)
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_dir"),  # TensorBoard 지원
    record_shapes=True,
    profile_memory=True,  # 메모리 사용량 기록
    with_stack=True  # 함수 호출 스택 정보 기록
) as prof:
    min_loss = 1000
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prof.step()
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