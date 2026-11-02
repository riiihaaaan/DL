# Deep Learning Lab Practices & Exam Solutions (BE IT LP-IV)

Practical implementations and exam solutions for Deep Learning (LP-IV). Solutions cover TensorFlow/Keras, PyTorch, and pure NumPy implementations with embedded execution outputs, evaluation metrics, and explanatory notes.

---

## Notebooks Index

| # | Notebook | Focus / Architecture | Framework |
|---|----------|----------------------|-----------|
| 01 | [01_Tensorflow_and_Pytorch_Basics.ipynb](01_Tensorflow_and_Pytorch_Basics.ipynb) | Tensors, operations, autograd, Keras diabetes classifier | PyTorch & TensorFlow |
| 02 | [02_Feedforward_Neural_Network_MNIST.ipynb](02_Feedforward_Neural_Network_MNIST.ipynb) | Dense feedforward network, Dropout, multiclass classification | Keras / TF |
| 03 | [03_Feedforward_Neural_Network_CIFAR.ipynb](03_Feedforward_Neural_Network_CIFAR.ipynb) | 3072-dim MLP, BatchNormalization, Dropout | Keras / TF |
| 04 | [04_CNN_Fashion_MNIST.ipynb](04_CNN_Fashion_MNIST.ipynb) | Conv2D, MaxPooling2D, Flatten, Dense | Keras / TF |
| 05 | [05_CNN_Pneumonia_XRay.ipynb](05_CNN_Pneumonia_XRay.ipynb) | Chest X-ray binary classification (NORMAL vs PNEUMONIA) | Keras / TF |
| 06 | [06_CNN_Food_Classification.ipynb](06_CNN_Food_Classification.ipynb) | Multiclass image classification on Food-101 | Keras / TF |
| 07 | [07_CNN_Brain_Tumor.ipynb](07_CNN_Brain_Tumor.ipynb) | 4-class MRI scan classification (glioma, meningioma, pituitary, no tumor) | Keras / TF |
| 08 | [08_RNN_Numpy_From_Scratch.ipynb](08_RNN_Numpy_From_Scratch.ipynb) | SimpleRNN class, Forward pass, BPTT, and gradient descent | Pure NumPy |
| 09 | [09_Autoencoder_Sparse_MNIST.ipynb](09_Autoencoder_Sparse_MNIST.ipynb) | Autoencoder with L1 activity regularizer bottleneck | Keras / TF |
| 10 | [10_Autoencoder_Anomaly_Detection_CreditCard.ipynb](10_Autoencoder_Anomaly_Detection_CreditCard.ipynb) | Credit card fraud detection via reconstruction loss thresholding | Keras / TF |
| 11 | [11_Autoencoder_Denoising_MNIST.ipynb](11_Autoencoder_Denoising_MNIST.ipynb) | Image denoising with Gaussian noise degradation | Keras / TF |
| 12 | [12_Transfer_Learning_Object_Detection_Food.ipynb](12_Transfer_Learning_Object_Detection_Food.ipynb) | MobileNetV2 feature extractor with bounding box regression head | Keras / TF |
| 13 | [13_Transfer_Learning_Animal_Classification.ipynb](13_Transfer_Learning_Animal_Classification.ipynb) | Pretrained MobileNetV2 on Cats vs Dogs (>90% accuracy) | Keras / TF |
| 14 | [14_Continuous_Bag_of_Words_CBOW.ipynb](14_Continuous_Bag_of_Words_CBOW.ipynb) | CBOW Word2Vec architecture, context averaging, cosine similarity | PyTorch |

---

## Quick Reference

- **Exam Cheat Sheet**: [EXAM_CHEAT_SHEET.md](EXAM_CHEAT_SHEET.md) — Concise code templates and conceptual summaries for all 14 assignments designed for quick revision.

---

## Requirements

- Python 3.9+
- TensorFlow >= 2.10
- PyTorch >= 2.0
- NumPy, Pandas, Scikit-Learn, Matplotlib
