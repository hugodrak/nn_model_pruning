import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from big_model import BigMNISTModel

def train_big_model(epochs=5, batch_size=64, lr=0.001):
    # Data Loading
    train_loader = DataLoader(
        datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST(
            root='./data',
            train=False,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Model, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BigMNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        test_loss, accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), "big_model.pth")

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return total_loss, accuracy

if __name__ == "__main__":
    train_big_model()