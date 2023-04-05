import pickle
import random

import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from bert import get_model


device = th.device("cpu")

batch_size = 256
num_epoch = 10
learning_rate = 1e-2
model_name = "bert-base-chinese"

base_model = get_model(model_name).to(device)
predictor = nn.Conv1d(768, 1, 3).to(device)
optimizer = optim.SGD(predictor.parameters(), lr=learning_rate, momentum=0.5)
loss_fn = nn.BCEWithLogitsLoss()

with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

random.seed(0)
random.shuffle(data)
print(data[0].sentence)
test_count = 50000
validation_count = 10000
data = data[:-test_count]

train_data = data[:-validation_count]
valid_data = data[-validation_count:]
print("train:", len(train_data))
print("valid:", len(valid_data))


def update(batch):
    total_loss = sum(x[0] for x in batch)
    total_weight = sum(x[1] for x in batch)
    loss = total_loss / total_weight
    loss.backward()
    optimizer.step()
    batch.clear()


for epoch in range(num_epoch):
    random.shuffle(train_data)
    batch = []
    for record in tqdm(train_data):
        if len(batch) % batch_size == 0:
            if batch:
                update(batch)
            optimizer.zero_grad()
        x = th.tensor([record.index]).to(device)
        y = th.tensor([[record.label]]).type(th.float32).to(device)
        with th.no_grad():
            hidden = base_model(x).last_hidden_state.transpose(2, 1)
        prediction = predictor(hidden)
        loss = loss_fn(prediction, y)
        weight = y.sum().item()
        batch.append((loss * weight, weight))

    update(batch)

    th.save(predictor, f"checkpoint/conv1d-all-{epoch:03d}.pkl")

    validation_loss = []
    validation_weight = 0
    with th.no_grad():
        for record in valid_data:
            x = th.tensor([record.index]).to(device)
            y = th.tensor([[record.label]]).type(th.float32).to(device)
            hidden = base_model(x).last_hidden_state.transpose(2, 1)
            prediction = predictor(hidden)
            weight = y.sum().item()
            loss = loss_fn(prediction, y).item() * weight
            validation_loss.append(loss)
            validation_weight += weight
        print(epoch, np.mean(validation_loss) / validation_weight)
