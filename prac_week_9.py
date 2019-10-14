
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.utils import shuffle


import matplotlib.pyplot as plt

def ex1_2():
    '''
    Save the arrays x_train, y_train, x_test and y_test
    into a single npz file named 'mnist_dataset.npz'   
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    #For faster training during the practival, reduce the number of examples
    # x_train, y_train = shuffle(x_train, y_train, random_state=0)
    # x_test, y_test = shuffle(x_test, y_test, random_state=0)
    
    # x_train = x_train[:30000]
    # y_train = y_train[:30000]
    # x_test = x_test[:3000]
    # y_test = y_test[:3000]
    
    img_rows, img_cols = x_train.shape[1:3]
    num_classes = len(np.unique(y_train))
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    batch_size = 128
    #for debugging use 3 epochs
    epochs = 3
    # epochs = 12

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
    
    #Create the model
    input_layer = keras.layers.Input(input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape)(input_layer)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01))(x)
    x = tf.nn.dropout(0.25)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.models.Model(input_layer, output)
    
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
    # Plot training & validation accuracy values
    # ------------INSERT YOUR CODE HERE-----------
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    # ------------INSERT YOUR CODE HERE-----------
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
                  
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
                             
def ex3():
    '''
    Read back the arrays x_train, y_train, x_test and y_test
    from the npz file named 'mnist_dataset.npz'.
    Then, print the shape and dtype of these numpy arrays.
    '''
    def triplet_loss(y_true, y_pred, margin = 0.4):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
    
        anchor = y_pred[0]
        positive = y_pred[1]
        negative = y_pred[2]
    
        # squared distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        # squared distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
        # compute loss
        basic_loss = margin + pos_dist - neg_dist
        return tf.reduce_mean(tf.maximum(basic_loss,0.0))
    
    #Test implementation of triplet loss function 
    num_data = 10
    feat_dim = 6
    margin = 0.2
    
    embeddings = [np.random.rand(num_data, feat_dim).astype(np.float32),
                  np.random.rand(num_data, feat_dim).astype(np.float32),
                  np.random.rand(num_data, feat_dim).astype(np.float32)]
    labels = np.random.randint(0, 1, size=(num_data)).astype(np.float32)
      
    #Compute loss with numpy
    loss_np = 0.
    anchor = embeddings[0]
    positive = embeddings[1]
    negative = embeddings[2]
    

    for i in range(num_data):
        pos_dist = np.sum(np.square(anchor[i] - positive[i]))
        neg_dist = np.sum(np.square(anchor[i] - negative[i]))
        loss_np += max(margin + pos_dist - neg_dist,0.0)
    loss_np /= num_data
    print('Triplet loss computed with numpy', loss_np)
    
    # Compute the loss in TF
    loss_tf = triplet_loss(labels, embeddings, margin)
    with tf.Session() as sess:
        loss_tf_val = sess.run(loss_tf)
        print('Triplet loss computed with tensorflow', loss_tf_val)
    assert np.allclose(loss_np, loss_tf_val)
    

if __name__ == '__main__':
  #ex1_2()
   ex3()  