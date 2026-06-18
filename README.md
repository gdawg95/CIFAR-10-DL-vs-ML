# CIFAR-10 Image Classification: From Classical ML to Deep Learning
 
A comparative study of image classification approaches on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, progressing from traditional machine learning baselines to state-of-the-art transfer learning. Built as part of a university Machine Learning unit. The project demonstrates how feature extraction (HOG, grayscale pixel vectors) compares against learned representations (CNNs, pretrained ImageNet models) on the same 10-class image classification task — and quantifies the accuracy gains at each step.
 
---
 
## Results Summary
 
| Model | Approach | Feature Extraction | Test Accuracy |
|---|---|---|---|
| Random Forest | Classical ML | HOG (4×4 cells) | 42.97% |
| SVM (RBF kernel) | Classical ML | HOG + GridSearchCV | 62.74% |
| Baseline CNN | Deep Learning | Learned (3-block ConvNet) | 75.57% |
| ResNet-18 | Transfer Learning | ImageNet pretrained, full fine-tune | 94.65% |
| EfficientNet-B0 | Transfer Learning | ImageNet pretrained, full fine-tune | **96.75%** |
 
The jump from SVM (62.7%) to a baseline CNN (75.6%) illustrates the advantage of learned hierarchical features over hand-crafted descriptors. Transfer learning then pushes accuracy from 75.6% to 96.8%, demonstrating how pretrained representations from ImageNet generalise effectively to a smaller, lower-resolution dataset.

## Repository Structure
 
```
├── notebooks/
│   │   └── ML_models.ipynb     # Random Forest, Logistic Regression, SVM
│   │   └── DL_models.ipynb     # Baseline CNN, ResNet-18, EfficientNet-B0
├── .gitignore
└── README.md
```
---
 
## Approach
 
### Classical ML Pipeline
 
All three classical models follow a shared pipeline: load CIFAR-10, extract features, train a classifier, and evaluate on a held-out test set.
 
**Feature extraction** is the critical bottleneck for classical methods on image data. Raw pixel values carry too much noise and redundancy for traditional classifiers to learn from efficiently, so we experimented with two strategies:
 
- **HOG (Histogram of Oriented Gradients):** Converts each image to grayscale, then computes gradient orientation histograms over local cells. This captures edge and shape information while discarding colour and absolute intensity — making it well-suited for object recognition tasks. Used with both Random Forest and SVM.
- **Grayscale pixel features:** A simpler baseline where each image is converted to a 1024-dimensional grayscale vector (32×32), normalised to [0, 1]. Used with Logistic Regression to establish a minimal-preprocessing baseline.
**Hyperparameter tuning** was performed using GridSearchCV with stratified cross-validation for both Logistic Regression (regularisation strength C) and SVM (C and gamma), ensuring the tuning set never leaked into final training data.
 
### Deep Learning Pipeline
 
The deep learning notebook implements three architectures of increasing complexity, all sharing a common training and evaluation framework to ensure fair comparison. 

**Data augmentation** was applied uniformly across models: random horizontal flips, colour jitter, random rotations, and random cropping with padding. For pretrained models (ResNet-18, EfficientNet-B0), images were resized to 224×224 and normalised using ImageNet channel statistics. The baseline CNN operated on native 32×32 inputs with normalisation computed from the CIFAR-10 training set.
 
**Baseline CNN:** A 3-block ConvNet (Conv → BatchNorm → ReLU → MaxPool) expanding from 32 to 128 filters, followed by a fully connected classifier with dropout regularisation. Trained from scratch to establish what a simple learned representation can achieve without pretrained knowledge.
 
**ResNet-18:** Pretrained on ImageNet with all layers unfrozen for end-to-end fine-tuning. The final fully connected layer was replaced to output 10 classes. A low learning rate (1e-4) was used to preserve pretrained features while allowing adaptation to CIFAR-10.
 
**EfficientNet-B0:** Same transfer learning strategy as ResNet-18 — pretrained ImageNet weights, full fine-tuning, classifier head replacement. EfficientNet's compound scaling (depth, width, and resolution balanced together) yields a more parameter-efficient architecture that outperformed ResNet-18 by ~2% on test accuracy.
 
---
 
## Dataset
 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 colour images (32×32 pixels) across 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The official split provides 50,000 training and 10,000 test images, with 6,000 images per class. We further split the training set 80/20 into training and validation partitions using stratified sampling.
 
The dataset was not included in this repository due to its size. It is downloaded automatically by `tensorflow.keras.datasets.cifar10` in the classical ML notebook, or loaded from local pickle files in the deep learning notebooks.

---

## Setup
 
**Classical ML notebook** requires scikit-learn, scikit-image, numpy, matplotlib, and seaborn. TensorFlow/Keras is used only for dataset loading.
 
**Deep Learning notebooks** require PyTorch, torchvision, scikit-learn, numpy, matplotlib, seaborn, and Pillow. Training was performed on Google Colab with GPU acceleration.
 
```bash
# Classical ML dependencies
pip install scikit-learn scikit-image matplotlib seaborn tensorflow
 
# Deep Learning dependencies
pip install torch torchvision scikit-learn matplotlib seaborn pillow
```

## Key Takeaways
 
**Feature extraction is the bottleneck for classical methods.** HOG improved SVM accuracy from raw-pixel baselines by capturing gradient structure, but hand-crafted features fundamentally limit what classical classifiers can learn from complex images.
 
**Transfer learning is disproportionately effective.** EfficientNet-B0 has ~4M parameters trained on 40,000 CIFAR-10 images (a ratio of ~0.01 samples per parameter), yet achieves 96.8% accuracy. This is only possible because ImageNet pretraining provides strong general-purpose visual features that transfer across domains.
 
**The generalisation gap tells a story.** Random Forest achieved 100% training accuracy but only 43% test accuracy (a 57-point gap) — a textbook case of overfitting to high-dimensional features without regularisation. EfficientNet showed a much smaller gap (~2%), indicating that transfer learning, data augmentation, and architectural design together produce models that generalise well.
 
---

## Contributors
 
This project was completed collaboratively by a team of six as part of a university Machine Learning unit.
 
**Classical ML Branch**
 
| Contributor | Model | Key Contributions |
|---|---|---|
| Nicholas | Random Forest | HOG feature extraction pipeline, accuracy and confusion matrix evaluation |
| Dylan | Logistic Regression | Grayscale pixel features, GridSearchCV hyperparameter tuning |
| Kayla | SVM (RBF kernel) | HOG + SVM pipeline, GridSearchCV for C and gamma |
 
**Deep Learning Branch**
 
| Contributor | Model | Key Contributions |
|---|---|---|
| Hai | Baseline CNN | Custom dataset class, data augmentation pipeline, code integration and refactoring |
| Vinay | ResNet-18 | ResNet implementation, training loop, test evaluation function, history tracking |
| Su | EfficientNet-B0 | EfficientNet implementation, validation function, checkpoint saving, learning curve visualisation, advanced fine-tuning experiments |
 
---
