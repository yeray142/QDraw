![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?style=for-the-badge&logo=Qiskit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/PyThon-3670A0.svg?style=for-the-badge&logo=Python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)

<p align="center">
    <br>
    <a href="https://example.com/">
        <img src="https://i.imgur.com/wGBaBlS.png" width=200>
    </a>
    <h3 align="center">QDraw: Quantum Draw</h3>
    <p align="center">
        Google's Quick Draw reimagination using quantum computing.
        <br>
        <a href="https://github.com/yeray142/QDraw/issues/new?template=bug.md">Report bug</a>
        ·
        <a href="https://github.com/yeray142/QDraw/issues/new?template=feature.md&labels=feature">Request feature</a>
    </p>
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#installation">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Introduction

> [!IMPORTANT]  
> This project is not finished yet, bugs are expected.

QDraw leverages the power of quantum computing through the Qiskit framework, in conjunction with the robust machine learning capabilities of PyTorch, to provide an advanced sketch recognition system. This project aims to push the boundaries of what's possible in the realm of digital sketch analysis by harnessing the parallel processing and unique problem-solving capabilities of quantum algorithms.

## Features
> [!WARNING]  
> The following is a provisional list of features for QDraw. Please note that these features are subject to change as the project evolves.
- [x] **Quantum-Enhanced ML:** Enhances the efficiency and performance of sketch recognition by combining quantum algorithms with deep learning models.
- [x] **Qiskit integration:** Employs the versatile Qiskit framework for quantum computing operations, placing QDraw at the forefront of quantum-assisted applications.
- [x] **PyTorch support:** Utilizes PyTorch for its dynamic computation graph and extensive library support, ensuring smooth integration between classical and quantum computing elements.
- [ ] **Real-time processing:** Capable of handling real-time sketch recognition tasks with improved speed and accuracy, thanks to the quantum computing backend.
- [ ] **User-friendly interface:** Offers an intuitive interface that allows users to easily draw, submit sketches, and receive instant recognition feedback.

## Getting Started

To get started with QDraw, follow these simple setup instructions.

### Prerequisites

- Python 3.8 or later (tested on 3.8)
- PyTorch
- Qiskit
- Rust (for compiling the Rust components)

### Installation

1. Clone the repository and navigate to the QDraw directory:
```
git clone https://github.com/yeray142/QDraw.git && cd QDraw
```
2. Install the required Python packages:
```
pip install -r requirements.txt
```
3. Navigate to the Rust directory within QDraw and compile the Rust components using Maturin. This step requires that you have Rust and Maturin installed:
```
cd rust && maturin develop --release
```
After completing these steps, you should have QDraw set up and ready to use on your system.

### Running QDraw

To explore QDraw's capabilities, head over to the `notebooks/` directory where you'll find Jupyter notebooks prepared for experimentation where you can replicate the experiments and results. 

Select a Jupyter notebook from the list to start experimenting with QDraw's features interactively.

## Contributing

We welcome contributions to QDraw! If you have suggestions for improvements or want to contribute code, please follow these steps:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
