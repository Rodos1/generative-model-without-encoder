import torch
import torch.nn
from optuna_scripts.d2nn_utilities import configure_dnn
from utilities.filters import Window, Gaussian
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def generate_noise(batch_size, size):
    noise = torch.randn(batch_size, 1, size, size)
    noise = abs(noise)/torch.max(abs(noise))
    return noise

#Гиперпараметры
n = 32
pixels = 32
length = 0.001
wavelength = 500E-9
masks = 3
distance = 0.08628599497985633
lr = 0.02
batch_size = 32

spectral_filter = Window(centers=wavelength, sizes=300.0E-9)

detectors_filter = Gaussian((length / 50, length / 50), (0, 0))

model = configure_dnn(n=n, pixels=pixels, length=length, wavelength=wavelength, masks_amount=masks, distance=distance, detectors_norm='none')

transform = transforms.Compose([
    transforms.Resize((n,n)),
    transforms.ToTensor()
])
dataset = datasets.MNIST(root='./data', train=True, transform = transform, download=True)
#dataset = [(img, label) for img, label in dataset if label == 5]  #Фильтруем только пятёрки
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs_num = 10
for epoch in range(epochs_num):
    total_loss = 0.
    for images, _ in dataloader:
        images = images.to(torch.float32)
        noise = generate_noise(images.shape[0], n)

        optimizer.zero_grad()
        outputs = model.forward(noise, detect=False)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch: [{epoch+1}/{epochs_num}], Loss: {total_loss/len(dataloader):.6f}')

import matplotlib.pyplot as plt

with torch.no_grad():
    generated_image_1 = model(generate_noise(1,n), detect=False).cpu().squeeze()
    generated_image_2 = model(generate_noise(1,n), detect=False).cpu().squeeze()
    generated_image_3 = model(generate_noise(1,n), detect=False).cpu().squeeze()
    generated_image_4 = model(generate_noise(1,n), detect=False).cpu().squeeze()
    generated_image_5 = model(generate_noise(1,n), detect=False).cpu().squeeze()

fig, axes = plt.subplots(1,5, figsize=(15,3))
plt.imshow(generated_image_1)

axes[0].imshow(generated_image_1)
axes[1].imshow(generated_image_2)
axes[2].imshow(generated_image_3)
axes[3].imshow(generated_image_4)
axes[4].imshow(generated_image_5)

plt.show()
