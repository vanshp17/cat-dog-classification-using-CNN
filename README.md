# Cat and Dog Classification using CNN

## Introduction
This project involves building Convolutional Neural Networks (CNN) for classifying images of cats and dogs. The dataset used can be found [here](https://www.kaggle.com/datasets/salader/dogs-vs-cats/data). To access the dataset, you need to create an API token from your Kaggle account.

## Getting Started

### Prerequisites
Make sure you have the required dependencies installed. You can use the following commands to set up your environment:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
```

### Installation
Download and unzip the dataset using the Kaggle API:

```bash
!kaggle datasets download -d salader/dogs-vs-cats
```

## Usage

### Input
The input data is divided into batches using generators. The images are loaded and processed using TensorFlow and Keras libraries.

```python
# generators- divide data into batches
train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)
```

### Output
The output is a binary classification indicating whether the image contains a cat or a dog.

## Training
The project includes the training of three different models:

1. **Model 1**: A basic CNN model with increasing complexity.
2. **Model 2**: Enhanced with Batch Normalization and Dropout layers to reduce overfitting.
3. **Model 3**: Further improvements with data augmentation.

```python
# Model 1
model1 = Sequential()
# ... (model architecture)
model1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model1.fit(train_ds, epochs=10, validation_data=validation_ds)
```

## Results
Each model's performance is evaluated and visualized using accuracy and loss plots.

```python
import matplotlib.pyplot as plt

# Accuracy Plot
plt.title('Accuracy')
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()

# Loss Plot
plt.title('Loss')
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()
```

## Contributing
Feel free to contribute to the project by suggesting improvements or additional features. Fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- Thanks to Kaggle for providing the dataset.
- Acknowledgments to the TensorFlow and Keras communities for their valuable libraries.
