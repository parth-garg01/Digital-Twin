import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _iterate_samples(inputs, targets):
    for x, y in zip(inputs, targets):
        yield x, y


def train_gnn_model(
    model,
    train_inputs,
    train_targets,
    edge_index,
    epochs=20,
    learning_rate=0.01,
    grad_clip=1.0,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    losses = []

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        sample_count = 0

        for x, y in _iterate_samples(train_inputs, train_targets):
            pred = model(x, edge_index)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_loss += float(loss.item())
            sample_count += 1

        losses.append(epoch_loss / max(1, sample_count))

    return losses


def train_sequence_model(
    model,
    train_inputs,
    train_targets,
    epochs=20,
    learning_rate=0.01,
    grad_clip=1.0,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    losses = []

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        sample_count = 0

        for x, y in _iterate_samples(train_inputs, train_targets):
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_loss += float(loss.item())
            sample_count += 1

        losses.append(epoch_loss / max(1, sample_count))

    return losses


def evaluate_predictions(predictions, targets):
    pred = np.asarray(predictions, dtype=np.float32)
    truth = np.asarray(targets, dtype=np.float32)
    return float(np.mean(np.abs(pred - truth)))
