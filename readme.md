# Documentation for AirbusSegmentation

## Data

The dataset used for this project is sourced from the [Airbus Ship Detection](https://www.kaggle.com/competitions/airbus-ship-detection) Kaggle competition.
This dataset comprises high-resolution images where ships of various sizes and shapes may be present. 
Each image is accompanied by an annotation file containing the coordinates of ship regions.


## Model

### Encoder Comparison

During the development of this project, various encoder architectures were evaluated for 
constructing the U-Net model for semantic segmentation. The architectures considered included the pure 
U-Net, VGG19, ResNet34, EfficientNetB7, and ResNet50. After thorough experimentation and analysis, 
it was found that ResNet50 outperformed the other architectures in terms of the dice score metric.



### Encoder (Downsampling)
1. Base Model: The U-Net model uses a ResNet50 base model pre-trained on the ImageNet dataset to
extract high-level features from input images. This base model is initialized with pre-trained weights.

2. Skip Connections: Skip connections are established from intermediate layers of the base model 
to the corresponding layers in the decoder. These skip connections help in preserving spatial information 
during the upsampling process.

### Decoder (Upsampling)
1. Upsampling Blocks: The decoder consists of a series of upsampling blocks. 
Each upsampling block consists of a convolutional layer followed by upsampling to increase 
the spatial resolution.

2. Concatenation with Skip Connections: At each stage of the decoder, the output from the corresponding encoder layer is concatenated with the output of the current decoder layer. This concatenation helps in combining low-level features with high-level features, aiding in better localization.

### Output Layer
1. Final Convolutional Layer: The final layer of the decoder is a convolutional layer that produces 
the segmentation mask. This layer uses a kernel size of 3x3 and applies transposed convolution 
(also known as deconvolution) to upsample the feature maps.

2. Activation Function: Sigmoid activation is applied to the output of the final convolutional layer 
to generate pixel-wise probabilities indicating the presence of the target class (e.g., ship or background).

### Model Initialization and Pre-trained Weights
The model is initialized with the specified input shape.
Optionally, pre-trained weights can be loaded into the model if a path to pre-trained weights is provided.
This U-Net model architecture effectively learns to segment ships from input images, 
leveraging both low-level and high-level features extracted from the ResNet50 base model.

### Training

The training block contains the implementation of the custom loss function `dice_bce_loss`, which combines the Dice coefficient and binary cross-entropy loss.

Additionally, the block includes the setup of the learning rate scheduler using the `PolynomialDecay` function to dynamically adjust the learning rate during training. The Adam optimizer is utilized with the configured learning rate scheduler.

Training callbacks are defined to save the model weights using `ModelCheckpoint` at each epoch if the validation loss improves, ensuring that the best model weights are saved.

## Evaluation

- Original Dataset: **Dice score** ≈ **0.77**
- Dataset with 90% Images Without Ships: **Dice score** ≈ **0.83**

## Usage instruction

Before using the model, ensure that all necessary libraries are installed in your environment. Use the `requirements.txt` file to specify the required library versions.

Install the libraries using pip:

```bash
pip install -r requirements.txt
```

### Training

To initiate model training, run the following command in your terminal:

```bash
python train.py
```

### Inference

To perform inference using the trained model, you can use the provided inference script. By default, if no arguments are provided, the script will run with the following flags: `-s -e -show`. 

However, you can still customize the behavior using the following optional arguments:

- **Evaluation Mode**: Use the `-e` or `--evaluate` flag to enable evaluation mode, which evaluates the model's performance on the dataset.
- **Prediction Mode**: Use the `-p` or `--predict` flag to enable prediction mode, which generates predictions for the input images.
- **Show Predictions**: Use the `-show` flag to display some of the generated predictions.

Additionally, you can specify the paths to the input images, CSV file with targets, and the output CSV file using the following optional arguments:

- `-split` flag to evaluate on a subset of the dataset, typically 10%.
- `--img` or `--images_path`: Path to the directory containing input images (default: 'imgs').
- `--csv` or `--csv_path`: Path to the CSV file containing targets (default: 'train_ship_segmentations_v2.csv').
- `--out` or `--output_path`: Path to the output CSV file for predictions (default: 'submission.csv').

To run the inference script with default flags, execute the following command in your terminal:

```bash
python inference.py [options]
```

## Author information

Oleksandr Shyshcuk

email: shyshchuko@gmail.com