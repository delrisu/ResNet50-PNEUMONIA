import numpy as np
import cv2
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD , RMSprop
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import skimage

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape=(150, 150, 3), classes=2):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    modelResNet -- a modelResNet() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create modelResNet
    modelResNet = Model(inputs=X_input, outputs=X, name='ResNet50')

    return modelResNet

def percents(x,pos):
    return '%1.0f' % (x*100)

def vgg():
    modelVGG = Sequential()
    modelVGG.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(200, 200, 3)))
    modelVGG.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    modelVGG.add(MaxPooling2D(pool_size=(2, 2)))

    modelVGG.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(200, 200, 3)))
    modelVGG.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    modelVGG.add(MaxPooling2D(pool_size=(2, 2)))

    modelVGG.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    modelVGG.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    modelVGG.add(MaxPooling2D(pool_size=(2, 2)))

    modelVGG.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    modelVGG.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    modelVGG.add(MaxPooling2D(pool_size=(2, 2)))

    modelVGG.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    modelVGG.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    modelVGG.add(MaxPooling2D(pool_size=(2, 2)))

    modelVGG.add(Flatten())

    modelVGG.add(Dense(64, activation=swish_activation))
    modelVGG.add(Dropout(0.4))
    modelVGG.add(Dense(2, activation='sigmoid'))

    modelVGG.compile(loss='binary_crossentropy',
                   optimizer=RMSprop(lr=0.00005),
                   metrics=['accuracy'])
    return modelVGG;

def swish_activation(x):
    return (K.sigmoid(x) * x)

def drawingPlot(data,filename):
    fig = plt.figure()
    plot = fig.add_subplot(111)
    plot.grid(False)
    plot.set_ylim(0, 1)
    plot.yaxis.set_major_formatter(FuncFormatter(percents))
    plot.set_xticks(range(2))
    plot.set_xticklabels(['zdrowy', 'chory'])
    plot.set_ylabel("Przypasowanie [%}")
    this_plot = plot.bar(range(2), data, color="#777777")
    this_plot[0].set_color('blue')
    this_plot[1].set_color('red')
    fig.savefig(filename)
    plt.clf()
def main():
    global modelResNet
    global modelVGG
    global imagePanel
    global ResNetPanel
    global VGGPanel
    global statusResNet
    global statusVGG
    modelVGG = vgg();
    modelVGG.load_weights("weightsVGG.hdf5")

    modelResNet = ResNet50(input_shape = (200, 200, 3), classes = 2)
    modelResNet.load_weights('weightsResNet.hdf5')

    m = Tk()

    img = Image.open('./przyklad.jpeg')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)

    wynikiResNet =Image.open('./wynikiResNet.jpg')
    wynikiResNet = wynikiResNet.resize((350, 350), Image.ANTIALIAS)
    wynikiResNet = ImageTk.PhotoImage(wynikiResNet)

    wynikiVGG = Image.open('./⁮wynikiVGG.jpg')
    wynikiVGG = wynikiVGG.resize((350, 350), Image.ANTIALIAS)
    wynikiVGG = ImageTk.PhotoImage(wynikiVGG)


    m.title("Wykrywanie zapalenia płuc")
    titleImage = Label(m, text="Zdjęcie rentgenowskie")
    titleImage.grid(row=0,column=1,pady=2,sticky=W+E)
    imagePanel = Label(m, image=img)
    imagePanel.grid(row=1,column=1, pady=2,sticky=W+E)

    titleResnet = Label(m, text="Wynik sieci ResNet")
    titleResnet.grid(row=0,column=2,pady=2, sticky=W+E)
    ResNetPanel = Label(m, image=wynikiResNet)
    ResNetPanel.grid(row=1, column=2, pady=2, sticky=W + E)
    statusResNet = Label(m, text="Detekcja zapalenia płuc")
    statusResNet.grid(row=2, column=2, pady=2, sticky=W+E)

    titleVGG = Label(m, text="Wynik sieci VGG")
    titleVGG.grid(row=0, column=3, pady=2, sticky=W + E)
    VGGPanel = Label(m, image=wynikiVGG)
    VGGPanel.grid(row=1, column=3, pady=2, sticky=W+E)
    statusVGG = Label(m, text="Detekcja zapalenia płuc")
    statusVGG.grid(row=2, column=3, pady=2, sticky=W+E)

    w = Button(m, text="Wybierz zdjęcie", command=ask_file_name)
    w.grid(row=2,column=1,pady=2,sticky=W)
    m.mainloop()

def ask_file_name():
    global modelResNet
    global modelVGG
    global imagePanel
    global ResNetPanel
    global VGGPanel
    global statusResNet
    global stutusVGG
    filepath=""
    filepath=askopenfilename()
    if(filepath!=""):
        print(filepath)
        img =Image.open(filepath)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        imagePanel.configure(image=img)
        imagePanel.image = img

        openCVimage = cv2.imread(filepath)
        if openCVimage is not None:
            openCVimage = skimage.transform.resize(openCVimage, (200, 200, 3))
            openCVimage = np.asarray(openCVimage)
        predictionsResNet = modelResNet.predict(np.array([openCVimage,]))[0]
        drawingPlot(predictionsResNet,'wynikiResNet.jpg')

        if(predictionsResNet[0]>0.75):
            statusResNet.configure(text="Jestes zdrowy!")
            statusResNet.text = "Jestes zdrowy!"
        elif(predictionsResNet[1]>0.75):
            statusResNet.configure(text="Jestes chory!")
            statusResNet.text = "Jestes chory!"
        else:
            statusResNet.configure(text="Nie wiadomo co Ci jest!")
            statusResNet.text = "Nie wiadomo co Ci jest!"

        wynikiResNet = Image.open('./wynikiResNet.jpg')
        wynikiResNet = wynikiResNet.resize((350, 350), Image.ANTIALIAS)
        wynikiResNet = ImageTk.PhotoImage(wynikiResNet)

        ResNetPanel.configure(image=wynikiResNet)
        ResNetPanel.image=wynikiResNet

        predictionsVGG=modelVGG.predict(np.array([openCVimage,]))[0]
        drawingPlot(predictionsVGG,'⁮wynikiVGG.jpg')

        if (predictionsVGG[0] > 0.75):
            statusVGG.configure(text="Jestes zdrowy!")
            statusVGG.text = "Jestes zdrowy!"
        elif (predictionsVGG[1] > 0.75):
            statusVGG.configure(text="Jestes chory!")
            statusVGG.text = "Jestes chory!"
        else:
            statusVGG.configure(text="Nie wiadomo co Ci jest!")
            statusVGG.text = "Nie wiadomo co Ci jest!"

        wynikiVGG = Image.open('./⁮wynikiVGG.jpg')
        wynikiVGG = wynikiVGG.resize((350, 350), Image.ANTIALIAS)
        wynikiVGG = ImageTk.PhotoImage(wynikiVGG)

        VGGPanel.configure(image=wynikiVGG)
        VGGPanel.image = wynikiVGG
    else:
        statusResNet.configure(text="Nie wybrano pliku!")
        statusResNet.text = "Nie wybrano pliku!"
        statusVGG.configure(text="Nie wybrano pliku!")
        statusVGG.text = "Nie wybrano pliku!"

if __name__=="__main__":
    main()
