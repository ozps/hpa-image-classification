import os, sys
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")
print(os.listdir('../input'))

SIZE = 512

# Load dataset info
path_to_train = '../input/train/'
data = pd.read_csv('../input/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')): 
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

# Create generator
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)  
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        
        image = np.stack((
        np.array(image_red_ch), 
        np.array(image_green_ch), 
        np.array(image_blue_ch),
        np.array(image_yellow_ch)), -1)
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

from sklearn.model_selection import train_test_split

# Split data into train, valid
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=20)

batch_size = 16

# Create train and valid datagens
train_generator = data_generator.create_train(
    train_dataset_info[train_indexes], batch_size, (SIZE, SIZE, 4), augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[valid_indexes], 32, (SIZE, SIZE, 4), augument=False)
    
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.applications.xception import Xception
from keras import metrics
from keras.optimizers import Adam 
from keras.models import Model

# Model    
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = Xception(include_top=False,
                   weights='imagenet', 
                   input_shape=(253, 253, 3),
                   pooling='avg', classes=28)
    x = BatchNormalization()(input_tensor)
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(3, kernel_size=(3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = base_model(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model

model = create_model(
    input_shape=(SIZE, SIZE, 4), 
    n_out=28)
    
# Train all layers
for layer in model.layers:
    layer.trainable = True
    
model.summary()
    
model.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])
            
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=7, 
    verbose=1)
    
# Create submit
submit = pd.read_csv('../input/sample_submission.csv')
predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = data_generator.load_image(path, (SIZE, SIZE, 4))/255.
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=0.25]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)