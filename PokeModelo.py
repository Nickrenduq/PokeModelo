import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

IMG_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Fuerza 3 canales (evita problemas con PNG con transparencia)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dos carpetas distintas: una para entrenar y otra para probar
train_data = datasets.ImageFolder(
    root="PokeModelo/pokemon_dataset/pokemon_survivors",
    transform=transform
)
test_data = datasets.ImageFolder(
    root="PokeModelo/poketest",   # <-- pon aquí tu carpeta de prueba
    transform=transform
)

# Las dos carpetas deben tener los mismos pokémon (mismos nombres de subcarpeta)
assert train_data.class_to_idx == test_data.class_to_idx, \
    f"Las clases no coinciden:\ntrain: {train_data.class_to_idx}\ntest: {test_data.class_to_idx}"

num_classes = len(train_data.classes)
print("Clases:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self, num_classes, img_size=64, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        reduced = img_size // 4
        self.fc1 = nn.Linear(32 * reduced * reduced, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = ConvNet(num_classes, img_size=IMG_SIZE, in_channels=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento (solo con la carpeta de entrenamiento)
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Evaluación (con la carpeta de prueba, imágenes distintas)
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Precisión en prueba: {100 * correct / total:.2f}%')