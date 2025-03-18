import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import time


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate, activation_function, batch_norm):
        super().__init__()
        layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_layers[i]))
            else:
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(activation_function)
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


data = pd.read_csv("data_PCT_AR.csv").dropna()
features = data[['HF', 'LF', 'LFHF', 'SCR', 'SCL','SKT', 'BMI', 'AR']]
labels = data['Comfort_Level']

X = features.values
y = labels.values

input_size = 8
hidden_layers = [403,287]
output_size = 5
num_epochs = 187
learning_rate = 0.001
dropout_rate = 0.4
weight_decay = 0.001
activation_function = nn.ReLU()
batch_norm = True
loss_function = nn.CrossEntropyLoss()
batch_size = 172

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kfold = KFold(n_splits=10, shuffle=True)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f'\nFold {fold + 1}/{kfold.n_splits}')

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = NeuralNetwork(input_size, hidden_layers, output_size, dropout_rate, activation_function, batch_norm).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_time = time.time()
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(test_loss)


    fold_accuracy = accuracy_score(all_targets, all_predictions)
    fold_results.append(fold_accuracy)
    torch.save(model.state_dict(), 'PCT_EDAandHRV.pth')
    print("Model saved successfully.")

    print(f'Fold {fold + 1} Accuracy: {fold_accuracy:.4f}')
    print(f'Classification Report:\n{classification_report(all_targets, all_predictions)}')
    print(f'Confusion Matrix:\n{confusion_matrix(all_targets, all_predictions)}')
    print(f'Total Time (seconds): {time.time() - start_time:.4f}')

print(f'\nAverage Accuracy across folds: {sum(fold_results) / len(fold_results):.4f}')
