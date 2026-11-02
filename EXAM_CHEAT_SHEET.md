# Lab Practice IV (Deep Learning) - Practical Exam Quick Study Guide

This cheat sheet summarizes the **exact, verified implementations** and **key concepts** for all 14 practical assignments directly referenced from the BE IT LP-IV Lab Manual and exam problem statements.

---

## 1. Universal 4-Step Pipeline for Keras Models
Every neural network in TensorFlow / Keras follows this exact flow:
1. **Load & Preprocess Data:** Scale pixels by dividing by `255.0` (or `Rescaling(1./255)` layer).
2. **Define Architecture:** `model = tf.keras.Sequential([ ... ])`.
3. **Compile Model:** `model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`.
4. **Fit & Evaluate:** `model.fit(X_train, y_train, epochs=..., batch_size=...)` then `model.evaluate(X_test, y_test)`.

---

## Summary of All 14 Practical Assignments

### Assignment 1: Basic Operations in TensorFlow and PyTorch
- **PyTorch:**
  - Creation: `torch.tensor(data)`, `torch.from_numpy(arr)`, `torch.zeros()`, `torch.ones()`, `torch.rand()`.
  - Math: `a + b`, `a * 2`, `torch.matmul(a, b)`, `a.T`, in-place `a.add_(10)`.
  - Autograd: `w = torch.tensor(2.0, requires_grad=True); y = 3*w**2 + 2*x + 1; y.backward(); print(w.grad)`.
- **TensorFlow:**
  - Creation: `tf.constant()`, `tf.Variable()` (for trainable weights), `tf.zeros()`, `tf.random.normal()`.
  - Math: `tf.add()`, `tf.matmul()`, `tf.transpose()`, `tf.reduce_mean()`, `tf.reshape()`.
  - GradientTape: `with tf.GradientTape() as tape: y = ...; tape.gradient(y, [w, x])`.
- **Keras Demo:** 3-Layer Sequential model on `pima-indians-diabetes.csv` (`Dense(12) -> Dense(8) -> Dense(1, sigmoid)`).

---

### Assignment 2: Feedforward Neural Network on MNIST
- **Input:** 784 flattened pixels (28x28 grayscale) normalized to $[0, 1]$.
- **Architecture:** `Dense(256, relu) -> Dropout(0.2) -> Dense(128, relu) -> Dense(10, softmax)`.
- **Loss:** `sparse_categorical_crossentropy` (integer digit labels 0–9).
- **Evaluation:** Test Accuracy + `classification_report(y_test, y_pred)`.

---

### Assignment 3: Feedforward Neural Network on CIFAR-10
- **Input:** 3072 flattened features (32x32x3 RGB channels).
- **Architecture:** `Dense(512, relu) -> BatchNormalization() -> Dropout(0.25) -> Dense(256, relu) -> Dense(10, softmax)`.
- **Note:** Batch Normalization stabilizes training on natural color images.

---

### Assignment 4: CNN on Fashion MNIST
- **Input Reshape:** `X.reshape(-1, 28, 28, 1)` (Height, Width, Channel).
- **CNN Backbone:** `Conv2D(32, (3,3), relu) -> MaxPool(2,2) -> Conv2D(64, (3,3), relu) -> MaxPool(2,2)`.
- **Classifier Head:** `Flatten() -> Dense(64, relu) -> Dropout(0.25) -> Dense(10, softmax)`.

---

### Assignment 5: CNN on Pneumonia Chest X-Ray
- **Dataset Loader:** `image_dataset_from_directory('.../Pneumonia X Ray/train', image_size=(64, 64))`.
- **Normalization:** `Rescaling(1./255)` layer at input of Sequential model.
- **Output:** Binary/2-class classification (`Dense(2, softmax)` with `sparse_categorical_crossentropy`).

---

### Assignment 6: CNN on Food Dataset (Food-101)
- **Class Filtering:** Pass `class_names=['french_fries', 'hamburger', 'ice_cream', 'pizza', 'sushi']`.
- **Architecture:** `Rescaling(1./255) -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(64) -> Dense(num_classes, softmax)`.

---

### Assignment 7: Brain Tumor Classification using CNN
- **Classes:** 4 classes (`glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`).
- **Input:** Brain MRI scans loaded via `image_dataset_from_directory`.
- **Output:** `Dense(4, activation='softmax')`.

---

### Assignment 8: Recurrent Neural Network (RNN) from Scratch in NumPy
- **Reference:** Lab Manual Assignment No. 7 (Pages 50-54).
- **Forward:** $h_t = \text{sigmoid}(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$, $y_t = W_{ho} \cdot h_t + b_o$.
- **Backward (BPTT):** Error at time $t$ propagates back through $dh_{\text{next}}$, updating $dW_{ho}, dW_{hh}, dW_{xh}, db_h, db_o$.
- **Training:** Gradient descent loop updating weights with learning rate.

---

### Assignment 9: Sparse Autoencoder on MNIST
- **Principle:** Unsupervised reconstruction ($X \to X$).
- **Sparsity Penalty:** Add `activity_regularizer=tf.keras.regularizers.l1(1e-4)` to the 32-dim bottleneck layer.
- **Verification:** Measure average activation across latent neurons and verify percentage of zero/near-zero activations.

---

### Assignment 10: Autoencoder Anomaly Detection (Credit Card Fraud)
- **Reference:** Lab Manual Assignment No. 4 (Pages 32-39).
- **Key Strategy:** Train Autoencoder **ONLY on Normal transactions** (`Class == 0`) with MSE loss.
- **Detection:** Fraud transactions produce significantly higher reconstruction errors. Flag anomalies using a threshold (e.g. 95th percentile of normal reconstruction error).

---

### Assignment 11: Image Denoising using Autoencoders on MNIST
- **Noise Addition:** $X_{\text{noisy}} = \text{clip}(X_{\text{clean}} + 0.35 \times \mathcal{N}(0, 1), 0.0, 1.0)$.
- **Training Trick:** Input is $X_{\text{noisy}}$, Target is $X_{\text{clean}}$ (`model.fit(X_train_noisy, X_train_clean)`).
- **Result:** Autoencoder learns to remove noise and reconstruct clean digits.

---

### Assignment 12: Object Detection via Transfer Learning on Food Dataset
- **Pretrained Backbone:** `MobileNetV2(include_top=False, weights='imagenet')` with `base_model.trainable = False`.
- **Bounding Box Head:** `GlobalAveragePooling2D() -> Dense(64, relu) -> Dense(4, sigmoid)`.
- **Output:** Normalized coordinates `[ymin, xmin, ymax, xmax]` in range $[0, 1]$ trained with MSE loss.

---

### Assignment 13: Image Classification via Transfer Learning on Animal Dataset
- **Dataset:** Cats vs Dogs (`dataset/cats_and_dogs_filtered/train`).
- **Architecture:** Pretrained `MobileNetV2` (frozen) + `GlobalAveragePooling2D()` + `Dense(1, sigmoid)`.
- **Result:** Reaches >90% validation accuracy in just 2 epochs thanks to pretrained ImageNet features.

---

### Assignment 14: Continuous Bag of Words (CBOW) Model in PyTorch
- **Reference:** Lab Manual Assignment No. 5 (Pages 40-45).
- **Objective:** Given context words $[w_{t-1}, w_{t+1}]$, predict the middle target word $w_t$.
- **Architecture:**
  ```python
  class CBOW(nn.Module):
      def __init__(self, vocab_size, emb_dim=16):
          super().__init__()
          self.embed = nn.Embedding(vocab_size, emb_dim)
          self.linear = nn.Linear(emb_dim, vocab_size)
      def forward(self, x):
          return self.linear(self.embed(x).mean(dim=1))
  ```
- **Embedding Analysis:** Compute cosine similarity between related words (`sim = dot(v1, v2) / (|v1|*|v2|)`).
