import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import shap
sns.set_style({'font.family': 'Times New Roman'})


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

model = NeuralNetwork(input_size, hidden_layers, output_size, dropout_rate, activation_function, batch_norm)
model.load_state_dict(torch.load("PCT_EDAandHRV.pth"))
model.eval()

file_path = r"C:\Users\apapo\physiological_signal\pythonProject\.venv\PTC_for_AR\Data\data_PCT_AR.csv"
df = pd.read_csv(file_path).dropna()

features = ['HF', 'LF', 'LFHF', 'SCR', 'SCL','SKT', 'BMI', 'AR']
target = 'Comfort_Level'
X = df[features].round(2)
y = df[target]




def model_wrapper(data):
    data = torch.FloatTensor(data)
    logits = model(data)
    probabilities = torch.softmax(logits, axis=1)
    return probabilities.detach().numpy()


explainer = shap.Explainer(model_wrapper, X.values)
shap_values = explainer(X)

#Global
shap.plots.beeswarm(shap_values[:,:,0], max_display=6)
shap.plots.beeswarm(shap_values[:,:,1], max_display=6)
shap.plots.beeswarm(shap_values[:,:,2], max_display=6)
shap.plots.beeswarm(shap_values[:,:,3], max_display=6)
shap.plots.beeswarm(shap_values[:,:,4], max_display=6)

#Local
shap.plots.force(shap_values[0, :, 0], show=True, matplotlib=True)
shap.waterfall_plot(shap_values[0,:,0], max_display=6)
