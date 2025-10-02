from __future__ import print_function
from torchvision import datasets, transforms, utils
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import Net

# Train Phase transformations

train_transforms = A.Compose([
  A.HorizontalFlip(p=0.5),
  A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=25, p=0.5),
  A.CoarseDropout(
        max_holes=1,           # Drop only one large region
        max_height=16,         # Max size for a CIFAR-10 image (32x32)
        max_width=16,          # Max size for a CIFAR-10 image (32x32)
        min_height=16,          # Min size
        min_width=16,           # Min size
        fill_value=tuple([int(x * 255) for x in [0.4914, 0.4822, 0.4465]]),          # Fill with black
        p=0.3                  # 50% chance of applying
    ),
  A.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
  ToTensorV2(),
])

def transform_wrapper(pil_img):
    """
    Wraps the Albumentations pipeline for torchvision datasets.
    
    1. Converts PIL Image -> NumPy Array (Albumentations input format).
    2. Runs the Albumentations pipeline (using the 'image=' named argument).
    3. Returns the augmented PyTorch Tensor.
    """
    # Convert PIL Image to NumPy array (H x W x C)
    image_np = np.array(pil_img)
    
    # Run the Albumentations pipeline, passing the image as a named argument
    augmented = train_transforms(image=image_np)
    
    # Return the PyTorch Tensor
    return augmented['image']

# Test Phase transformations
test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Dataset and Creating Train/Test Split
train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_wrapper)
test = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

# Dataloader Arguments & Test/Train Dataloaders
SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

# # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
# train_data = train.train_data
# train_data = train.transform(train_data.numpy())

# print('[Train]')
# print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
# print(' - Tensor Shape:', train.train_data.size())
# print(' - min:', torch.min(train_data))
# print(' - max:', torch.max(train_data))
# print(' - mean:', torch.mean(train_data))
# print(' - std:', torch.std(train_data))
# print(' - var:', torch.var(train_data))

dataiter = iter(train_loader)
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)

# Let's visualize some of the images

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5     # Un-normalize the image from [-1, 1] to [0, 1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# Define the class names for CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Show images from the batch
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
imshow(utils.make_grid(images[:4]))

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(3, 32, 32))

# Training and Testing
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=10000.0
    )
    scheduler.step()
    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

# Train and test model
#from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR

model =  Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
EPOCHS = 60


for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

t = [t_items.item() for t_items in train_losses]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(t)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")