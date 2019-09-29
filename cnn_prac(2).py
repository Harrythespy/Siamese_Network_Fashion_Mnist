
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

def ex1():
    '''
    Save the arrays x_train, y_train, x_test and y_test
    into a single npz file named 'mnist_dataset.npz'   
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # INSERT YOUR CODE HERE TO SAVE x_train, y_train, x_test and y_test to a file named 'mnist_dataset.npz'
    np.savez('mnist_dataset.npz', x_train=x_train, y_train = y_train, x_test = x_test, y_test = y_test)

                             
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
    Then, display the training images indexed from 25 to 35 and their classes.

    '''
    with np.load('mnist_dataset.npz') as npzfile:   
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        pass
        
    for i in range(25,35):
        plt.imshow(x_train[i])     # INSERT YOUR CODE HERE     
        plt.title('{}'.format(y_train[i]))    # INSERT YOUR CODE HERE
        plt.show()
        

def ex4():
    '''
    Build, train and evaluate a CNN on the mnist dataset
    
    '''
    
    # Read back the arrays x_train, y_train, x_test and y_test
    # from the npz file named 'mnist_dataset.npz'.
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
    # epochs = 3 # debugging code

    # x_train = x_train[:1000]
    # y_train = y_train[:1000]


    # convert to float32 and rescale between 0 and 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    #
    # convert class vectors to binary class matrices (aka "sparse coding" or "one hot")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    
    # Create an object of Sequential()
    model = keras.models.Sequential()
    
    hidden_nodes = 32
    # Add layer 0 to layer 5
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128,activatio='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='Adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
            
    
    # Train the model by calling the fit method of the model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
              
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
#    ex1()
#    ex2()
#    ex3()
   ex4()    
