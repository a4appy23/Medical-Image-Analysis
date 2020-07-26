import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
import keras
from sklearn.model_selection import RepeatedKFold,cross_val_score,KFold
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from skimage import measure
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,Dense
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2DTranspose
from keras.regularizers import l2
from keras.layers import  Activation
from keras.callbacks import ReduceLROnPlateau
from sklearn import metrics
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras import optimizers

patch= glob.glob(r"C:\Users\a4app\Documents\medical image analysis\dataset\*patch*")
mask = glob.glob(r"C:\Users\a4app\Documents\medical image analysis\dataset\*mask*")
count=  glob.glob(r"C:\Users\a4app\Documents\medical image analysis\dataset\*count*")



im_width = 256
im_height = 256
border = 3


X_data = []
files = glob.glob (r"C:\Users\a4app\Documents\medical image analysis\dataset\*patch*")
for myFile in files:
    print(myFile)
    image = cv2.imread (myFile)
    image = np.array(image).astype('float32')
    image= resize(image, (256, 256, 3), mode = 'constant', preserve_range = True)
    X = image/255.0
    X_data.append (X)


y_data = []
files = glob.glob (r"C:\Users\a4app\Documents\medical image analysis\dataset\*mask*")
for myFile in files:
    print(myFile)
    image = cv2.imread (myFile)
    image = np.array(image).astype('float32')
    print("after",image.shape)
    y = image/255.0
    y_data.append (y)
    
X= np.array(X_data)
y= np.array(y_data)
 
print('X_data shape:', np.array(X_data).shape)
print('y_data shape:', np.array(y_data).shape)

for i in range(10):
    rkf = KFold(n_splits=5,random_state=26421,shuffle=True)
    for train_index, test_index in rkf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))


datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    samplewise_std_normalization=False,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.3
     )


datagen.fit(X_train)
datagen.fit(X_test)


def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

  return 1 - numerator / denominator

cvscores = []

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = MaxPooling2D((2, 2), (2, 2))(c) 
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    
    return c



def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((256, 256, 3))
    
    p0 = inputs
 
    c1, p1 = down_block(p0, f[0]) 
    BatchNormalization()(p1)
    c2, p2 = down_block(p1, f[1])
    BatchNormalization()(p2)
    c3, p3 = down_block(p2, f[2])
    BatchNormalization()(p3)
    c4, p4 = down_block(p3, f[3]) 
    BatchNormalization()(p4)
    
    
    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])
    BatchNormalization()(u1)
    u2 = up_block(u1, c3, f[2])
    BatchNormalization()(u2)
    u3 = up_block(u2, c2, f[1]) 
    BatchNormalization()(u3)
    u4 = up_block(u3, c1, f[0])
    BatchNormalization()(u4)    
    outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid",name='main_output')(u4)   
    model = Model(inputs, outputs)
    opt=Adam(lr=0.001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
    #model.compile(optimizer=opt, loss=dice_loss, metrics=["acc"])
    return model

model = UNet()


model.summary()

#model.compile(optimizer=opt, loss=dice_loss, metrics=["acc"])

callbacks = [   
    EarlyStopping(patience=10, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


history=model.fit(datagen.flow(X_train, y_train, batch_size=20),steps_per_epoch=(len(X_train)//20),validation_data=(X_test, y_test), epochs=10)
   
# fits the model on batches with real-time data augmentation:
print(y_train.shape)
#history=model.fit_generator(datagen.flow(X_train, y_train, batch_size=20), callbacks=callbacks,steps_per_epoch=(len(X_train)/20),validation_data=(X_test, y_test), epochs=30)


print(len(X_train))
preds_test = model.predict(X_test, verbose=1)
preds_val = model.predict(y_test, verbose=1)


for i in range(10):
    for train_index, test_index in rkf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        score=model.evaluate(X_test, y_test, batch_size=19)

        print("the evaluation score is==============")
        print(score)


        print("%s: %.2f%%" %(model.metrics_names[1], score[1]*100))
        cvscores.append(score[1] * 100)
    
    
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


        print((np.mean(cvscores), np.var(cvscores),np.std(cvscores)))
#print(len(X_train))




preds_test = model.predict(X_test, verbose=1)
preds_val = model.predict(y_test, verbose=1)




    
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()  

