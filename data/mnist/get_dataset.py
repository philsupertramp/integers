#!/usr/bin/env python3
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import struct
import os

os.makedirs('data/mnist', exist_ok=True)

print("Downloading MNIST via torchvision...")
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=ToTensor())

def write_idx_images(filename, images):
    """Write images as IDX3-ubyte format"""
    with open(filename, 'wb') as f:
        # Magic number for IDX3-ubyte
        f.write(struct.pack('>I', 0x00000803))
        # Dimensions: n_images, rows, cols
        f.write(struct.pack('>I', len(images)))
        f.write(struct.pack('>I', 28))
        f.write(struct.pack('>I', 28))
        # Data
        for img in images:
            # img is [1, 28, 28], convert to bytes [0, 255]
            img_bytes = (img.squeeze() * 255).byte().numpy()
            f.write(img_bytes.tobytes())
    print(f"✓ Written {filename}")

def write_idx_labels(filename, labels):
    """Write labels as IDX1-ubyte format"""
    with open(filename, 'wb') as f:
        # Magic number for IDX1-ubyte
        f.write(struct.pack('>I', 0x00000801))
        # Dimension: n_labels
        f.write(struct.pack('>I', len(labels)))
        # Data
        for label in labels:
            f.write(struct.pack('B', int(label)))
    print(f"✓ Written {filename}")

# Extract images and labels
train_images = torch.stack([img for img, _ in train_dataset])
train_labels = torch.tensor([label for _, label in train_dataset])
test_images = torch.stack([img for img, _ in test_dataset])
test_labels = torch.tensor([label for _, label in test_dataset])

# Write IDX format
write_idx_images('./train-images-idx3-ubyte', train_images)
write_idx_labels('./train-labels-idx1-ubyte', train_labels)
write_idx_images('./t10k-images-idx3-ubyte', test_images)
write_idx_labels('./t10k-labels-idx1-ubyte', test_labels)

print("\n✓ MNIST ready in data/mnist/")
print(f"  Train: {len(train_images)} images")
print(f"  Test:  {len(test_images)} images")
