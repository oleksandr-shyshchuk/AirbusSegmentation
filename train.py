import os

import cv2
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, ReLU, Input, Concatenate
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow import reduce_sum, round, clip_by_value
from tensorflow.keras.backend import epsilon

import warnings
warnings.filterwarnings('ignore')


def get_train_val_df(df, img_path):
    """
    Function to split a DataFrame into training and validation sets based on the presence of ships in images.

    Args:
        df (DataFrame): Input DataFrame containing information about images and ship presence.
        img_path (str): Path to the directory containing the images.

    Returns:
        tuple: A tuple containing two DataFrames - the training set and the validation set.
    """
    df = df[df['ImageId'].isin(os.listdir(img_path))]

    df['ShipCount'] = df.apply(lambda x: 0 if pd.isna(x['EncodedPixels']) else 1, axis=1)
    ships_numbers = df[['ImageId','ShipCount']].groupby(['ImageId']).sum()

    ships_numbers = ships_numbers[ships_numbers['ShipCount'] > 1]

    train_ids, valid_ids = train_test_split(ships_numbers,
                 test_size = 0.1,
                 stratify = ships_numbers['ShipCount'], random_state=42)

    df = df.drop(columns=['ShipCount'])
    train_df = df[df['ImageId'].isin(train_ids.index)]
    valid_df = df[df['ImageId'].isin(valid_ids.index)]
    return train_df, valid_df

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


class DataGenerator(Sequence):
    """
    A generator class to provide batches of images and corresponding masks for training a model.
    The class provides functionality to generate batches of images and masks, optionally applying data augmentation.
    """
    def __init__(self, images_dir, images_df, batch_size=16, image_size=(224, 224), augument=False):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.augument = augument

        self.all_batches = list(images_df.groupby('ImageId'))
        np.random.shuffle(self.all_batches)

    def __len__(self):
        return int(len(self.all_batches) / self.batch_size)

    def get_params(self, rotation_range=30):
        angle = np.random.uniform(-rotation_range, rotation_range)
        horizontal = np.random.random() < 0.5
        vertical = np.random.random() < 0.5

        return angle, horizontal, vertical

    def augment_image(self, image, params):
        angle, horizontal, vertical = params
        rotated_image = cv2.warpAffine(image,
                                       cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1),
                                       (image.shape[1], image.shape[0]))

        if horizontal:
            rotated_image = cv2.flip(rotated_image, 1)
        if vertical:
            rotated_image = cv2.flip(rotated_image, 0)

        return rotated_image

    def __getitem__(self, idx):
        batch_images = self.all_batches[idx * self.batch_size: (idx + 1) * self.batch_size]

        X = []
        y = []
        for batch_image in batch_images:
            image_path = os.path.join(self.images_dir, batch_image[0])

            # Load image and resize
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.image_size)

            # Load mask and resize
            mask = masks_as_image(batch_image[1]['EncodedPixels'])
            mask = cv2.resize(mask, self.image_size)
            mask = np.expand_dims(mask, axis=-1)

            if self.augument:
                params = self.get_params()
                image = self.augment_image(image, params)
                mask = self.augment_image(mask, params)

            image = image.astype(np.float32) / 255.0

            # as original image net
            image -= np.array([0.485, 0.456, 0.406])
            image /= np.array([0.229, 0.224, 0.225])

            mask = mask > 0.5

            X.append(image)
            y.append(mask)

        return np.array(X), np.array(y).astype(np.float32)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.all_batches)


class PredictDataGenerator(Sequence):
    """
    A generator class to provide batches of images for prediction.

    This class is similar to DataGenerator but does not require masks as output.
    """
    def __init__(self, images_dir, batch_size=8, image_size=(224, 224), shuffle=False):
        self.images = os.listdir(images_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.images_dir = images_dir
        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        add1 = (len(self.images) % self.batch_size) > 0
        return int(len(self.images) / self.batch_size) + add1

    def __getitem__(self, idx):
        if idx == (len(self)-1):
            end_id = None
        else:
            end_id = (idx + 1) * self.batch_size
        batch_images = self.images[idx * self.batch_size:end_id]

        X = []
        for batch_image in batch_images:
            image_path = os.path.join(self.images_dir, batch_image)

            # Load image and resize
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.image_size)

            image = image.astype(np.float32) / 255.0

            # as original image net
            image -= np.array([0.485, 0.456, 0.406])
            image /= np.array([0.229, 0.224, 0.225])

            X.append(image)

        return np.array(X), batch_images

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def upsample(filters, size):
    """
    Upsampling block for U-Net architecture.
    """
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same'))
    result.add(BatchNormalization())
    result.add(ReLU())
    return result


def unet_model(input_shape, weights_path=None):
    """
    Defines the U-Net model architecture for semantic segmentation.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        weights_path (str): Path to pre-trained weights (optional).

    Returns:
        Model: U-Net model for semantic segmentation.
    """
    # Load the ResNet50 base model with pre-trained ImageNet weights
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

    # Define layer names for skip connections
    layer_names = [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out',
    ]

    # Get output tensors of selected layers from the base model
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Define the downsampling part of the U-Net model
    down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)

    # Allow layers in the down_stack to be trainable
    down_stack.trainable = True

    # Define the upsampling part of the U-Net model using upsample blocks
    up_stack = [
        upsample(512, 3),
        upsample(256, 3),
        upsample(128, 3),
        upsample(64, 3)
    ]

    # Define the input tensor for the U-Net model
    inputs = Input(shape=input_shape)

    # Pass the input tensor through the downsampling part of the U-Net model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Perform upsampling and concatenation with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])

    # Final convolutional layer to generate segmentation mask
    last = Conv2DTranspose(
        filters=1, kernel_size=3, strides=2,
        padding='same')

    # Apply sigmoid activation
    x = last(x)
    x = sigmoid(x)

    model = Model(inputs=inputs, outputs=x)

    # Load pre-trained weights if provided
    if weights_path:
        model.load_weights(weights_path)

    return model


def dice_bce_loss(targets, inputs, smooth=1e-6):
    """
    Calculate the combined Dice coefficient and binary cross-entropy loss.

    Args:
        targets (tensor): Ground truth masks.
        inputs (tensor): Predicted masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tensor: Combined Dice coefficient and binary cross-entropy loss.
    """
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = binary_crossentropy(targets, inputs)
    intersection = K.sum(targets * inputs)
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


def dice_score(targets, inputs, smooth=1e-6):
    """
    Calculate the Dice coefficient.

    Args:
        targets (tensor): Ground truth masks.
        inputs (tensor): Predicted masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tensor: Dice coefficient.
    """
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(targets * inputs)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice


def dice_loss(targets, inputs, smooth=1e-6):
    """
    Calculate the Dice loss.

    Args:
        targets (tensor): Ground truth masks.
        inputs (tensor): Predicted masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tensor: Dice loss.
    """
    dice = dice_score(targets, inputs, smooth)
    return 1 - dice


def precision(y_true, y_pred):
    """
    Calculate precision metric.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.

    Returns:
        tensor: precision.
    """
    true_positives = reduce_sum(round(clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = reduce_sum(round(clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon())
    return precision


def recall(y_true, y_pred):
    """
    Calculate recall metric.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.

    Returns:
        tensor: Recall.
    """
    true_positives = reduce_sum(round(clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = reduce_sum(round(clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon())
    return recall


def get_callbacks():
    """
    Get a list of callback functions for training.

    Returns:
        list: List of callback functions.
    """
    model_checkpoint = ModelCheckpoint(filepath='checkpoint.keras',
                                       save_weights_only=True,
                                       save_best_only=True,
                                       verbose=1)
    return [model_checkpoint]


if __name__ == '__main__':
    csv_path = 'train_ship_segmentations_v2.csv'
    img_path = 'imgs'
    checkpoint_path = 'checkpoint'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    df = pd.read_csv(csv_path)
    train_df, valid_df = get_train_val_df(df, img_path)

    train_generator = DataGenerator(img_path, train_df, batch_size=8, augument=True)
    val_gen = DataGenerator(img_path, valid_df, batch_size=8)
    input_shape = (224, 224, 3)
    model = unet_model(input_shape=input_shape, weights_path='weights/model_weights.keras')

    metrics = [precision, recall]

    # Define learning rate schedule
    initial_learning_rate = 0.01

    decay_steps = 1000
    end_learning_rate = 0.00001
    power = 0.5

    lr_schedule = PolynomialDecay(
        initial_learning_rate,
        decay_steps,
        end_learning_rate,
        power)

    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss=dice_bce_loss,
                  metrics=metrics)

    model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10,
              callbacks=get_callbacks(),
              validation_data=val_gen, validation_steps=len(val_gen))

