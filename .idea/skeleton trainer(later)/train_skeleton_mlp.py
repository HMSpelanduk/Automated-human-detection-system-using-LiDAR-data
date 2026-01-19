import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skeleton_dataset import SkeletonDataset
from skeleton_mlp import SkeletonMLP

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SkeletonDataset(root_dir="skeleton_data")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SkeletonMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(20):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")

torch.save(model.state_dict(), "skeleton_mlp_classifier.pth")
print("Training complete. Model saved.")
