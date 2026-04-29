#!/usr/bin/env python3
"""
RemoteCUDA v3.0 — PyTorch Training Example
===========================================
Demonstrates how to use RemoteCUDA with PyTorch for
transparent remote GPU training.

This example shows:
    1. Installing CUDA hooks
    2. Defining a PyTorch model
    3. Training on remote GPU (or CPU fallback)
    4. No changes to standard PyTorch code!

Requirements:
    pip install remotecuda[client]
    (This installs remotecuda + PyTorch)

Usage:
    python pytorch_training.py
    python pytorch_training.py --server 192.168.1.100
    python pytorch_training.py --epochs 10
"""

import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='RemoteCUDA PyTorch Training Example')
    parser.add_argument('--server', default=None, help='Server IP')
    parser.add_argument('--port', type=int, default=55555, help='Server port')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    print("=" * 60)
    print("  RemoteCUDA v3.0 — PyTorch Training Example")
    print("=" * 60)
    print()

    # Import remotecuda
    import remotecuda

    # Connect
    print("1. Connecting to RemoteCUDA server...")
    if args.server:
        remotecuda.init(server=args.server, port=args.port)
    else:
        remotecuda.init()

    info = remotecuda.info()
    print(f"   Device: {info.get('device', 'unknown')}")
    print()

    # Install PyTorch CUDA hooks
    print("2. Installing PyTorch CUDA hooks...")
    from remotecuda.client.hook import install_hook

    install_hook()
    print("   Hooks installed. All .cuda() calls will use remote GPU.")
    print()

    # Import PyTorch
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Define a simple model
    print("3. Defining model...")
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    model = SimpleModel()
    # This .cuda() call is intercepted and sent to remote GPU!
    model = model.cuda()
    print(f"   Model: {sum(p.numel() for p in model.parameters())} parameters")
    print()

    # Create synthetic data
    print("4. Creating synthetic training data...")
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    print(f"   Samples: {len(dataset)}")
    print(f"   Batches: {len(dataloader)}")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print(f"5. Training for {args.epochs} epochs...")
    print()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(dataloader):
            # These .cuda() calls are intercepted!
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time

        print(f"   Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")

    print()

    # Evaluation
    print("6. Evaluating model...")
    model.eval()
    with torch.no_grad():
        X_test = torch.randn(100, 784).cuda()
        y_test = torch.randint(0, 10, (100,)).cuda()
        output = model(X_test)
        pred = output.argmax(dim=1)
        accuracy = (pred == y_test).float().mean().item()
        print(f"   Test accuracy: {accuracy:.2%}")

    print()

    # Cleanup
    print("7. Cleaning up...")
    from remotecuda.client.hook import uninstall_hook
    uninstall_hook()
    remotecuda.shutdown()
    print("   Done!")

    print()
    print("=" * 60)
    print("  Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()