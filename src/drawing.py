import numpy as np
from flask import Flask, request, jsonify, render_template
from qiskit_machine_learning.connectors import TorchConnector

import torch
import torch.nn.functional as F
from torch import cat
from torch.nn import Module, Conv2d, Linear, Flatten, BatchNorm2d

from src.circuit import create_qnn
from utils import *
from rust_sketch import sketches_to_images

app = Flask(__name__)


# Define torch NN module
class Net(Module):
    def __init__(self, qnn):
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

        # Hybrid dense layers
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=32 * 7 * 7, out_features=5)

        # Quantum layer
        self.qnn = TorchConnector(qnn)
        self.fc2 = Linear(1, out_features=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))  # Batch normalization ?? ReLU ??
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.fc1(self.flatten(x)))  # Batch normalization ??

        x = self.qnn(x)
        x = self.fc2(x)  # ReLU ?? Batch normalization ??
        return cat((x, 1 - x), -1)


@app.route('/')
def index():
    return render_template('canvas.html')


@app.route('/classify', methods=['POST'])
def classify():
    strokes = request.json['strokes']
    plot_doodle(strokes)
    print(strokes)
    imgs = np.array(sketches_to_images([strokes], 255, 28))
    image_array = np.array(imgs[0], dtype=np.uint8).reshape((28, 28))
    # image = Image.fromarray(image_array, 'L')
    # image.show()
    # print(image)

    # Convert preprocessed strokes to an image or directly to a tensor for your model
    # For simplicity, let's assume you have a function `strokes_to_tensor` for this
    qnn, _ = create_qnn(5)
    model = Net(qnn)
    model.load_state_dict(torch.load("../notebooks/models/model.pt"))

    # Use your PyTorch model for prediction
    input_tensor = torch.tensor(image_array, dtype=torch.float32).reshape(-1, 28, 28)
    print(input_tensor.unsqueeze(1).shape)
    model_output = model(input_tensor.unsqueeze(1))
    print(model_output)
    pred = model_output.argmax(dim=1, keepdim=True)
    print("{}".format("Camera" if pred.item() == 0 else "Cell Phone"))

    prediction = 'your doodle'
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
