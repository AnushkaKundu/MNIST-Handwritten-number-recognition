# Handwritten Digit Recognition using Convolutional Neural Networks (CNN)

This repository contains code for a Convolutional Neural Network (CNN) model that can recognize handwritten digits from the MNIST dataset. It provides a simple GUI interface where users can draw a digit on a canvas, and the model will predict the digit based on the input.

## Requirements

Make sure you have the following dependencies installed:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- TensorFlow (`tensorflow`)
- Keras (`keras`)

You can install these dependencies using `pip`:

```
pip install opencv-python numpy matplotlib tensorflow keras
```

## Dataset

The model is trained on the MNIST dataset, which is a widely used dataset for handwritten digit recognition. The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels.

The dataset is automatically downloaded and loaded using the `keras.datasets.mnist` module.

## Preprocessing

Before feeding the data into the model, some preprocessing steps are applied:

1. The images are reshaped to have a single channel (grayscale) and the shape (28, 28, 1).
2. The pixel values are normalized to the range [0, 1] by dividing them by 255.
3. The labels are one-hot encoded using `keras.utils.to_categorical`.

## Model Architecture

The model architecture consists of the following layers:

1. Convolutional layer with 32 filters, kernel size of (3, 3), ReLU activation, and 'same' padding.
2. Max pooling layer with pool size (2, 2).
3. Convolutional layer with 64 filters, kernel size of (3, 3), ReLU activation, and 'same' padding.
4. Max pooling layer with pool size (2, 2).
5. Dropout layer with a rate of 0.25.
6. Flatten layer to convert the 2D feature maps to 1D.
7. Dense layer with 128 units and ReLU activation.
8. Dropout layer with a rate of 0.5.
9. Dense layer with 10 units (corresponding to the 10 digit classes) and softmax activation.

The model is compiled with the following settings:

- Loss function: Categorical cross-entropy
- Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.01
- Evaluation metric: Accuracy

## Training

The model is trained with the following hyperparameters:

- Batch size: 32
- Number of epochs: 10

The training progress is displayed during the training process, showing the loss and accuracy on both the training and validation sets.

## Evaluation

After training, the model is evaluated on the test set to measure its performance. The test loss and accuracy are printed.

## Saving and Loading the Model

The trained model is saved in the file `MNIST_10_epochs.h5` using the `model.save()` method from Keras. The saved model can be loaded using `load_model()` from `tensorflow.keras.models`.

## GUI for Digit Recognition

The script provides a simple GUI interface using OpenCV, where users can draw a digit on a canvas. Pressing Enter will pass the drawn digit to the model for prediction. The predicted digit is then displayed on the canvas. Pressing 'c' will clear the canvas and allow drawing a new digit.

## How to Run

1. Install the required dependencies mentioned above.
2. Clone this repository.
3. Navigate to the repository directory.
4. Run the following command:

   ```
   python MNISTdigit.py
   ```

   This will open a GUI window

 where you can draw a digit.

5. Use the mouse to draw a digit on the canvas.
6. Press Enter to get the model's prediction for the drawn digit.
7. Press 'c' to clear the canvas and draw a new digit.
8. Press Esc to exit the GUI.

Note: Make sure you have a webcam connected to your system, as the script uses OpenCV for GUI functionality.

## License

This project is licensed under the [MIT License](LICENSE).

## Contribution

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

By contributing to this project, you agree to license your contributions under the terms of the MIT License.
