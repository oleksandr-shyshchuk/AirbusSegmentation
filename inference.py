import argparse
import os.path

import pandas as pd
import train
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='UResNet50 Inference')

# Додавання аргументів
parser.add_argument('-e', '-evaluate', action='store_true', help='Enable evaluation mode')
parser.add_argument('-split', action='store_true', help='Evaluate on 10% set')
parser.add_argument('-p', '-predict', action='store_true', help='Enable prediction mode')
parser.add_argument('-show', action='store_true', help='Show some predictions')

parser.add_argument('--img', '--images', '--images_path', metavar='PATH', type=str, help="Path to images",
                    default='imgs')
parser.add_argument('--csv', '--csv_path', metavar='PATH', type=str, help="Path to CSV file with targets",
                    default='train_ship_segmentations_v2.csv')
parser.add_argument('--out', '--output_path', metavar='PATH', type=str, help="Path to CSV file with targets",
                    default='submisssion.csv')

args = parser.parse_args()


def get_model():
    """
    Load the U-Net model for semantic segmentation.
    """
    weights_path = 'weights/model_weights.keras'
    model = train.unet_model((224, 224, 3), weights_path=weights_path)
    model.compile(metrics=[train.dice_score])
    return model


# ----------- Evaluate ------------


def get_df():
    """
    Load DataFrame containing image and mask information.
    """
    df = pd.read_csv(args.csv)
    df = df[df['ImageId'].isin(os.listdir(args.img))]
    if args.split:
        train_df, test_df = train.get_train_val_df(df, args.img)
        df = test_df
    return df


def evaluate(model, df):
    """
    Evaluate the model using the provided DataFrame.
    """
    print('Evaluating...')
    data_gen = train.DataGenerator(images_dir=args.img, images_df=df)

    dice_score = model.evaluate(data_gen)

    print('Dice score: ', dice_score[1])


# ------------ Predict ----------


def mask_to_rle(img, shape=(768, 768)) -> str:
    """
    :param img: numpy 2D array, 1 - mask, 0 - background
            shape: (height,width) dimensions of the image
    :return: run length encoded pixels as string formated
    """
    img = img.astype('float32')
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def encode_minibatch(outputs):
    """
    Encode predicted masks to RLE format for a minibatch of outputs.
    """
    encoded_list = []
    for i in range(len(outputs)):
        predicted_mask = cv2.resize(outputs[i], (768, 768))
        encoded_list.append(mask_to_rle(predicted_mask))

    return encoded_list


# --------- Show ----------
def show_predictions(model):
    """
    Generate and display predictions made by the model on a set of images.
    """
    datagen = train.PredictDataGenerator(args.img, shuffle=True)

    desired_samples = 16
    samples_collected = 0
    combined_data = []
    combined_imgs = []
    i = 0

    while samples_collected < desired_samples:
        batch = datagen[i][0]
        images_path = datagen[i][1]

        combined_data.extend(batch)
        combined_imgs.extend(images_path)

        samples_collected += len(combined_data)

    combined_data = np.array(combined_data[:desired_samples])

    masks = model.predict(combined_data)

    fig, axes = plt.subplots(8, 4, figsize=(8, 15))

    for i, ax in enumerate(axes.flat):
        idx = i // 2
        if i % 2 == 0:
            image = cv2.imread(os.path.join(args.img, combined_imgs[idx]))
            ax.imshow(image)
            ax.set_title('Image')
        else:
            ax.imshow(masks[idx], cmap='gray')
            ax.set_title('Mask')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('example.png')
    plt.show(block=False)
    plt.pause(5)


if __name__ == '__main__':
    model = get_model()

    all_modes = not (args.show and args.p and args.e)

    # Show predictions if specified
    if args.show or all_modes:
        show_predictions(model)

    # Evaluate model if specified
    if args.e or all_modes:
        df = get_df()
        evaluate(model, df)

    # Predict and encode masks if specified
    if args.p or all_modes:
        datagen = train.PredictDataGenerator(images_dir=args.img)
        data_iter = iter(datagen)
        print('Predicting...')

        for batch, images in tqdm.tqdm(data_iter, total=len(datagen)):
            predictions = model.predict(batch, verbose=0)

            rle_encoded = encode_minibatch(predictions)

            header = not os.path.exists(args.out)

            combined_array = [images, rle_encoded]

            submission = pd.DataFrame(combined_array, index=['ImagesId', 'EncodedPixels']).T

            if header:
                submission.to_csv(args.out, header=True, index=False)
            else:
                submission.to_csv(args.out, header=False, index=False, mode='a')
