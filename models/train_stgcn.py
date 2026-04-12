import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, data_loader, edge_index, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for t in range(len(data_loader.X) - 1):
            x = torch.tensor(data_loader.X[t][:, 0], dtype=torch.float32)
            y = torch.tensor(data_loader.X[t+1][:, 0], dtype=torch.float32)
            

            x = x.unsqueeze(1)  # (nodes, features)
            y = y.unsqueeze(1)

            pred = model(x, edge_index)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")