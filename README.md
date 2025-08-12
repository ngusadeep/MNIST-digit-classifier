# MNIST Digit Classifier

## Project Overview

This project implements a handwritten digit recognition system using the MNIST dataset. The dataset contains 70,000 grayscale images of handwritten digits (0 to 9), each sized 28x28 pixels.

Using Scikit-Learn's Stochastic Gradient Descent (SGD) classifier, the model is trained to classify images into the correct digit categories. This project demonstrates key concepts in data preprocessing, model training, evaluation, and prediction for image classification tasks.

## Installation

1. Ensure you have Python 3.7 or newer installed.

2. Install the required Python packages:

```bash
pip install numpy matplotlib scikit-learn python-mnist
```

````

3. Download the MNIST dataset files (train and test images and labels) from [the official source](http://yann.lecun.com/exdb/mnist/) or other trusted repositories.

4. Place the MNIST dataset files in a folder named `mnist` in your project directory, or update the path accordingly in your code.



## How to Use / Predict

1. **Load and normalize the data:**

```python
from mnist import MNIST
import numpy as np

mndata = MNIST('./mnist')  # Update path if needed
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Normalize pixel values to [0,1]
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)
```

2. **Train the SGD classifier:**

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
```

3. **Make predictions:**

```python
some_digit = X_test[0]  # change the index to predict other digits
prediction = sgd_clf.predict([some_digit])
print("Predicted digit:", prediction[0])
```

4. **Visualize the digit (optional):**

```python
import matplotlib.pyplot as plt

some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show()
```



## Notes

- The MNIST dataset is already split into training (60,000 samples) and testing (10,000 samples) sets.
- Normalization of pixel values helps improve model performance.
- You can replace `SGDClassifier` with other classifiers from Scikit-Learn as needed.



Feel free to extend this project by experimenting with different models, hyperparameters, or by implementing more advanced deep learning approaches!


````
