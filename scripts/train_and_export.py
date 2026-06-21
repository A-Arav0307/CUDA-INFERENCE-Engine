import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

training_data = datasets.MNIST(root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.MNIST(root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct

epochs = 5
for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)

accuracy = test(test_dataloader, model, loss_fn)
print(f"Final accuracy: {(100*accuracy):>0.1f}%")
print("Done!")

#saving weights to weights directory under scripts
weights_dict = model.state_dict()
#keys(['linear_relu_stack.0.weight', 'linear_relu_stack.0.bias', 'linear_relu_stack.2.weight', 'linear_relu_stack.2.bias'])
key_mapping = {
    'linear_relu_stack.0.weight': 'W1',
    'linear_relu_stack.0.bias':   'b1',
    'linear_relu_stack.2.weight': 'W2',
    'linear_relu_stack.2.bias':   'b2'
}

weights_dir = os.path.join("scripts", "weights") 
os.makedirs(weights_dir, exist_ok=True)
save_path = os.path.join(weights_dir, "model_weights.pth")
torch.save(model.state_dict(), save_path) 

for original_key, new_name in key_mapping.items():
    if original_key in weights_dict:
        numpy_array = weights_dict[original_key].detach().cpu().numpy()
        binary_path = os.path.join(weights_dir, f"{new_name}.bin")
        numpy_array.tofile(binary_path)
        print(f"Saved {original_key} -> {binary_path} (Shape: {numpy_array.shape})")

        
#export test sampples to cpu_math.cpp 


sample_loader = DataLoader(test_data, batch_size=100, shuffle=False)
test_images, test_labels = next(iter(sample_loader)) 

images_flattened = test_images.view(100, 784).numpy().astype(np.float32)
label_flattened = test_labels.numpy().astype(np.int64)
model.eval()
with torch.no_grad():
    logits = model(test_images.to(device))
    preds_npy = logits.argmax(dim=1).cpu().numpy().astype(np.int64)

output_path = os.path.join(weights_dir, "test_images.bin")
label_path = os.path.join(weights_dir, "test_labels.bin")
preds_path = os.path.join(weights_dir, "test_preds_pytorch.bin")

images_flattened.tofile(output_path)
label_flattened.tofile(label_path)
preds_npy.tofile(preds_path) 

