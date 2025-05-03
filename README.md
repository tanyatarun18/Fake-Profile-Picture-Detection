Fake-Profile-Picture-Detection

This project focuses on detecting "AI-generated (fake) profile pictures" using "image forensics and deep learning techniques". The approach involves converting input images to grayscale and applying "Laplacian filters" to highlight subtle artifacts typically left behind by "GAN-generated (fake) images".

A "ResNet18 convolutional neural network (CNN)" is trained to classify images as either "real" or "fake" based on these visual clues. The model leverages "ReLU activations", "cross-entropy loss", and is evaluated using metrics like "accuracy", "precision", "recall", "F1-score", and a "confusion matrix".

📂 Dataset

Dataset used:
🔗 [140K Real and Fake Faces Dataset – Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/data)
The dataset contains 70,000 real human faces and 70,000 fake (GAN-generated) faces, making it ideal for binary classification tasks in "image forensics".


### ✅ Features

* "Grayscale image conversion" for enhanced feature extraction.
* "Laplacian filter" to emphasize edges and visual inconsistencies.
* "CNN-based classification" using "ResNet18" for balanced accuracy and performance.
* "Evaluation with detailed classification metrics".
* Suitable for research in "deepfake detection", "forensic AI", and "social media safety".
