# This is a practice of normalization, regulization, initialization
# optimization, data augmentation to classify non_mnist data set


import keras.callbacks
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, Conv2D, MaxPooling2D, Dropout
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.initializers import lecun_normal, VarianceScaling
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from data import load_data

def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)
    num_feat_map = 16
    nb_classes = 10
    hidden_size = 128
    
    #keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
   
    inp = Input(shape=input_shape)
    cov1 = Conv2D(num_feat_map, kernel_size=(1, 5), activation='relu',padding='same')(inp)
    pool1 = MaxPooling2D(pool_size=(1, 2))(cov1)
    d1 = Dropout(0.5)(pool1)
    cov2 = Conv2D(num_feat_map, kernel_size=(1, 5), activation='relu',padding='same')(d1)
    pool2 = MaxPooling2D(pool_size=(1, 2))(cov2)
    d2 = Dropout(0.5)(pool2)
    f = Flatten()(d2)
    h_1 = Dense(hidden_size,kernel_initializer= lecun_normal(seed=None))(f)
    n_1 = BatchNormalization()(h_1)
    a_1 = Activation('relu')(n_1)
    h_2 = Dense(hidden_size, kernel_initializer= VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None), kernel_regularizer=regularizers.l2(0.001))(a_1)
    n_2 = BatchNormalization()(h_2)
    a_2 = Activation('relu')(n_2)
    out = Dense(nb_classes, activation='softmax')(a_2)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())
    return model

batch_size = 128
nb_epoch = 15

# Load data
(X_train, y_train, X_test, y_test) = load_data()

# Load and compile model
model = get_model()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)

model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test),callbacks=[keras.callbacks.EarlyStopping(monitor='val loss', patience=5)])

model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train) / 32, epochs=20)

score = model.evaluate(X_test, y_test, verbose=1)

print("Accuracy:", score[1])

