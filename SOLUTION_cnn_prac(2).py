'''
SOLUTIONS to ifn680 W08 prac, 2019

'''


import numpy as np

#from tensorflow.contrib import keras 
import tensorflow as tf
from tensorflow import keras


import matplotlib.pyplot as plt

def ex1():
    '''
    Save the arrays x_train, y_train, x_test and y_test
    into a single npz file named 'mnist_dataset.npz'   
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # INSERT YOUR CODE HERE TO SAVE x_train, y_train, x_test and y_test to a file named 'mnist_dataset.npz'
    np.savez('mnist_dataset.npz', x_train = x_train, y_train = y_train, 
                              x_test = x_test , y_test = y_test)
                              
                             
def ex2():
    '''
    Read back the arrays x_train, y_train, x_test and y_test
    from the npz file named 'mnist_dataset.npz'.
    Then, print the shape and dtype of these numpy arrays.
    
    '''
    with np.load('mnist_dataset.npz') as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']
    
    print('x_train : ', x_train.shape, x_train.dtype)
    print('y_train : ', y_train.shape, y_train.dtype)
    print('x_test : ', x_test.shape, x_test.dtype)
    print('y_test : ', y_test.shape, y_test.dtype)


def ex3():
    '''
    Read back the arrays x_train and y_train,
    from the npz file named 'mnist_dataset.npz'.
    Then, display the training images indexed from 25 to 35 and their classes
    
    '''
    with np.load('mnist_dataset.npz') as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
    
    for i in range(25,35):
        plt.imshow(x_train[i],cmap='gray')
        plt.title(str(y_train[i]))
        plt.show()


def ex4():
    '''
    Build, train and evaluate a CNN on the mnist dataset
    
    '''
    with np.load('mnist_dataset.npz') as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']
    
    img_rows, img_cols = x_train.shape[1:3]
    num_classes = len(np.unique(y_train))
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    batch_size = 128
    epochs = 12

    #    epochs = 3 # debugging code
    #    x_train = x_train[:8000]
    #    y_train = y_train[:8000]


    # convert to float32 and rescale between 0 and 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    #
    # convert class vectors to binary class matrices (aka "sparse coding" or "one hot")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
              
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

'''
Exercise 5:
  padding  +
  image pixel *
  n = 2*k  
  assume n = 6 = 2*3
  image:       ******
  padded:    +* ** ** *+
  We have  k+1 pixel pairs when sliding the window.
  Given there are two filters, we get 2*(k+1) = n+2 neurons
  in the hidden layer. 
  
Exercise 6:
  Answer : 2x2

'''

if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()    
