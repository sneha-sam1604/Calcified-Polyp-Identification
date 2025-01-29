# Calcified-Polyp-Identification
This repository contains a Convolutional Neural Network (CNN) model for binary image classification using Keras. It includes data augmentation, model training, validation, and saving of trained weights. The project is structured to classify images into two categories with improved generalization through real-time image transformations.

This project was completed in 2016.

# Image Classification Using Convolutional Neural Networks (CNN)

## Abstract
This project focuses on binary classification of images using a Convolutional Neural Network (CNN). The model is trained to distinguish between two classes based on features extracted through multiple convolutional layers. The training process incorporates image augmentation techniques to enhance generalization and prevent overfitting. 

The entire workflow is divided into the following key phases:
1. **Image Preprocessing & Augmentation**: Images are resized and transformed using rotation, width/height shifts, shearing, zooming, and horizontal flipping.
2. **Dataset Preparation**: The images are loaded from directories into batches using the `ImageDataGenerator` class.
3. **CNN Model Creation**: A sequential CNN model is constructed with multiple convolutional, activation, pooling, flattening, and dense layers.
4. **Model Compilation**: The model is compiled using binary cross-entropy as the loss function and RMSprop as the optimizer.
5. **Training & Validation**: The model is trained using the augmented dataset, and validation is performed on a separate set of images.
6. **Performance Evaluation**: The validation accuracy and loss metrics are analyzed to assess model performance.
7. **Model Saving**: The trained model weights are saved for future use.

## Features
- **Data Augmentation**: Enhances dataset variability to improve generalization.
- **CNN-based Feature Extraction**: Uses convolutional layers to extract meaningful features from images.
- **Binary Classification**: Employs a sigmoid activation function for distinguishing between two classes.
- **Training with Keras**: Utilizes `fit_generator` for model training with real-time augmentation.
- **Model Persistence**: Saves trained weights for further analysis and use.

## Dataset Structure
The images should be organized in the following format:
```
project_root/
│── train/
│   ├── class_1/
│   ├── class_2/
│── validation/
│   ├── class_1/
│   ├── class_2/
```
- `train/`: Contains training images categorized into subfolders by class.
- `validation/`: Contains validation images organized similarly.

## Installation & Setup
1. Install the required dependencies:
   ```bash
   pip install tensorflow keras numpy matplotlib
   ```
2. Place your dataset inside the `train/` and `validation/` directories following the structure above.
3. Run the training script to train the model:
   ```bash
   python train.py
   ```

## Model Architecture
The CNN consists of:
- **Convolutional Layers**: Three Conv2D layers with ReLU activation.
- **Pooling Layers**: MaxPooling2D to reduce spatial dimensions.
- **Fully Connected Layers**: Dense layers with dropout to prevent overfitting.
- **Final Output Layer**: A single neuron with sigmoid activation for binary classification.

## Training the Model
The model is compiled with:
- **Loss Function**: Binary Crossentropy
- **Optimizer**: RMSprop
- **Metric**: Accuracy

It is trained using `fit_generator` with real-time data augmentation:
```python
model.fit_generator(
    train_generator,
    steps_per_epoch=13 // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=160 // batch_size
)
```

## Saving the Model
After training, the model weights are saved as:
```python
model.save_weights('50_epochs.h5')
```

## Future Improvements
- Implement **early stopping** to prevent overfitting.
- Use **Adam optimizer** instead of RMSprop for better performance.
- Add **more convolutional layers** to improve feature extraction.
- Save the **entire model**, not just weights, for easier reusability.
