import pandas as pd
import matplotlib.pyplot as plt

from qiskit_algorithms.utils import algorithm_globals
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import no_grad, manual_seed
from torch.nn import Module, Conv2d, Linear, Flatten, BatchNorm2d
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rust_sketch import sketches_to_images

SEED = 42

BATCH_SIZE = 32
N_SAMPLES = 100
LEARNING_RATE = 0.001
N_EPOCHS = 100

manual_seed(SEED)
algorithm_globals.random_seed = SEED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



camera_df = pd.read_json('../data/quickdraw/full_simplified_camera.ndjson', lines=True, nrows=2000)
phone_df = pd.read_json('../data/quickdraw/full_simplified_cell phone.ndjson', lines=True, nrows=2000)
calculator_df = pd.read_json('../data/quickdraw/full_simplified_calculator.ndjson', lines=True, nrows=2000)

# Drop unnecessary columns and process drawings
for df in [camera_df, phone_df, calculator_df]:
    df.drop(["timestamp", "countrycode", "key_id", "recognized"], axis=1, inplace=True)
    df["drawing"] = sketches_to_images(df.drawing.array, 255, 28)

# Concatenate DataFrames
df = pd.concat([camera_df, phone_df, calculator_df])


# Split data into train and test sets for camera data
X_train, X_test, y_train, y_test = train_test_split(
    df['drawing'], df['word'], test_size=0.2, random_state=SEED)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# Train Dataset
# --------------
X_train_tensor = torch.tensor(X_train.tolist(), dtype=torch.float32).reshape(-1, 28, 28).to(device)
y_train_tensor = torch.tensor(y_train.tolist(), dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test Dataset
# --------------
X_test_tensor = torch.tensor(X_test.tolist(), dtype=torch.float32).reshape(-1, 28, 28).to(device)
y_test_tensor = torch.tensor(y_test.tolist(), dtype=torch.long).to(device)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


from qiskit.circuit import ParameterVector

def quantum_feature_map(n: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n)
    features = ParameterVector("x", n)

    for i in range(n):
        circuit.rx(features[i], i)

    return circuit


def quantum_ansatz(n: int) -> QuantumCircuit:
    # Creamos un circuito cuántico con n qubits.
    circuit = QuantumCircuit(n)
    weights = ParameterVector("w", n * 3)

    # Asumiendo que 'weights' es una lista de valores de peso,
    # y la longitud es suficiente para aplicar Rz a cada qubit.
    for i in range(n):
        circuit.rz(weights[i], i)

    # Aplicar más rotaciones, que en la imagen parecen ser Ry
    for i in range(n):
        circuit.ry(weights[n + i], i)

    for i in range(n):
        circuit.rz(weights[2 * n + i], i)

    # Aplicar entrelazamiento con puertas CNOT
    for i in range(n - 1):
        circuit.cx(i, i + 1)
    circuit.cx(n - 1, 0)

    return circuit

# Define and create QNN
def create_qnn(n):
    feature_map = quantum_feature_map(n)
    ansatz = quantum_ansatz(n)

    qc = QuantumCircuit(n)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn, qc

qnn, qc = create_qnn(5)

from torch.nn import BatchNorm1d


# Define torch NN module
class Net(Module):
    def __init__(self, qnn, num_classes=2):
        super().__init__()
        # Classical convolutional layer
        self.conv1 = Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            padding=2
        )
        self.bn1 = BatchNorm2d(16)

        self.conv2 = Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.bn2 = BatchNorm2d(32)

        # Hybrid dense layers
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=32 * 7 * 7, out_features=5)
        self.bn3 = BatchNorm1d(5)

        # Quantum layer
        self.qnn = TorchConnector(qnn)
        self.fc2 = Linear(1, out_features=num_classes)
        self.bn4 = BatchNorm1d(num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn3(self.fc1(self.flatten(x))))

        x = self.qnn(x)
        x = F.relu(self.bn4(self.fc2(x)))

        # Apply softmax for multi-class classification
        x = F.softmax(x, dim=1)
        return x


model = Net(qnn, num_classes=3).to(device)

from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np

# Define optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = CrossEntropyLoss()

# Start training
epochs = N_EPOCHS
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

model.train()

with open('training_log_senokosov_3class.txt', 'w') as f:
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for data, target in tqdm(train_loader, desc="Training Batches", leave=False):
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            output = model(data.unsqueeze(1))

            # Calculate loss
            loss = loss_func(output, target)
            loss.backward()

            # Optimize weights
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss_list.append(total_loss / len(train_loader))
        train_acc_list.append(100 * correct / total)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.unsqueeze(1))
                loss = loss_func(output, target)
                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_loss_list.append(val_loss / len(test_loader))
        val_acc_list.append(100 * val_correct / val_total)

        model.train()

        # Print to file instead of console
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss_list[-1]:.4f} "
              f"Train Acc: {train_acc_list[-1]:.2f}% "
              f"Val Loss: {val_loss_list[-1]:.4f} "
              f"Val Acc: {val_acc_list[-1]:.2f}%", file=f)

# Save the lists to .npz file
np.savez('training_data_senokosov_3class.npz', train_loss=np.array(train_loss_list), train_acc=np.array(train_acc_list),
         val_loss=np.array(val_loss_list), val_acc=np.array(val_acc_list))


# Plotting the metrics
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.savefig('loss_over_epochs_senokosov3class.png')  # Save the plot to a file

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_acc_list, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig('accuracy_over_epochs_senokosov3class.png')  # Save the plot to a file

plt.close()  # Close the figure


torch.save(model.state_dict(), "model_senokosov3class.pt")
