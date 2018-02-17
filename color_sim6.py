import jsonlines
# from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, InputLayer, MaxPool2D, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant
import json
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
from PIL import Image


# Load data into numpy arrays
def load_data(data_file_path):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    with jsonlines.open(data_file_path, 'r') as infile:
        for item in infile:
            img_path = '/Users/jdo/dev/scrapers/scraper10/scraper10/spiders/images/full/' + item['img_hash'] + '.jpg'
            with open(img_path, 'r+b') as f:
                with Image.open(f) as picture:
                    # img = image.load_img(img_path, target_size=(128, 128), grayscale=False)
                    pic = picture.resize((128, 128), Image.ANTIALIAS)
                    # x = image.img_to_array(img)
                    x = np.array(pic)
                    x = (x - 255) / 255
                    x = np.expand_dims(x, axis=0)
                    # x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

                    train_prob = np.random.random()*10

                    item_data1 = json.loads(item['img_features'])
                    y = item_data1

                    print('hash: ', str(item['img_hash']))
                    if train_prob > 1:
                        X_train.append(x)
                        Y_train.append(y)
                    else:
                        X_test.append(x)
                        Y_test.append(y)

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


X_train, Y_train, X_test, Y_test = load_data('/Users/jdo/dev/colorsim/uniq_items1.jsonl')

# Reshape from array to vector
Y_train = Y_train.reshape(Y_train.shape[0], 1000)

Y_test = Y_test.reshape(Y_test.shape[0], 1000)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], X_train.shape[4])

X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3], X_test.shape[4])

print('X train shape: ')
print(X_train.shape)

print('Y train shape: ')
print(Y_train.shape)

print('X test shape: ')
print(X_test.shape)

print('Y test shape: ')
print(Y_test.shape)

model = Sequential()

model.add(
    InputLayer(input_shape=(128, 128, 3))
)

model.add(
    BatchNormalization()
)

# Next step is to add convolution layer to model.
model.add(
    Conv2D(
        64, (3, 3),
        padding='same',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform'
    )
)
# Add max pooling layer for 2D data.
model.add(MaxPool2D(padding='valid'))

# Next step is to add convolution layer to model.
model.add(
    Conv2D(
        128, (3, 3),
        padding='same',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform'
    )
)
# Add max pooling layer for 2D data.
model.add(MaxPool2D(padding='valid'))

# Next step is to add convolution layer to model.
# model.add(
#     Conv2D(
#         256, (3, 3),
#         padding='same',
#         bias_initializer=Constant(0.01),
#         kernel_initializer='random_uniform'
#     )
# )
# Add max pooling layer for 2D data.
model.add(MaxPool2D(padding='valid'))

model.add(Flatten())

# model.add(Dense(units=4096, activation='relu', kernel_initializer='random_uniform'))

# model.add(Dense(units=2048, activation='relu', kernel_initializer='random_uniform'))

model.add(Dense(units=1000, activation='relu', kernel_initializer='random_uniform'))

model.add(Dense(units=1000, activation='softmax', kernel_initializer='random_uniform'))

# model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, Y_train, epochs=20, batch_size=32)

evaluation = model.evaluate(x=X_test, y=Y_test)

print(evaluation)

model.save('colorsim_model10.h5')
