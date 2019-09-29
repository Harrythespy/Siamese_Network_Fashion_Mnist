import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def saveDataset():
    '''
      1. Download the dataset from Fashion Mnist of keras datasets 
      2. Gather the images and labels of both training and testing together
      3. Distribute them with the given conditions (80%-20%) and remove unnecessary classes
    '''
    # Download the dataset from keras.fashion
    fashion_mnist = keras.datasets.fashion_mnist
    # give names for the downloaded data
    (train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

    #put training data and testing data together and them separate them as 80-20 with given condition
    imgs = np.concatenate((train_imgs, test_imgs))
    labels = np.concatenate((train_labels, test_labels))
    print(imgs.shape)
    print(labels.shape)
    
    #split training dataset into training and validating
    # Set 1 is the images with labels for classes "top", "trouser", "pullover", "coat", "sandal", "andke boot"
    labels1 = [0, 1, 2, 4, 5, 9]
    
    # Set 2 is the images with labels for classes "dress", "sneaker", "bag", "shirt"
    labels2 = [3, 6, 7, 8]
    
    # Create new dataset in order to store different classes of images and labels
    set1_imgs = []
    set1_labels = []
    set2_imgs = []
    set2_labels = []
    
    # Classify both images and labels with different classes
    for data in range(0, len(imgs)):
      if labels[data] in labels1:
        set1_imgs.append(imgs[data])
        set1_labels.append(labels[data])
      else:
        set2_imgs.append(imgs[data])
        set2_labels.append(labels[data])           
    
    # Since the initial type of the arrays are python arrays
    # Convert into numpy arrays
    set1_imgs = np.array(set1_imgs)
    set1_labels = np.array(set1_labels)
    set2_imgs = np.array(set2_imgs)
    set2_labels = np.array(set2_labels)
    
    # Distribute training and testing datasets with 80/20 percent
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(set1_imgs, set1_labels, train_size=0.8, random_state=87)
#     print('x_train : ', train_imgs.shape, train_imgs.dtype)
#     print('y_train : ', train_labels.shape, train_labels.dtype)
#     print('x_test : ', test_imgs.shape, test_imgs.dtype)
#     print('y_test : ', test_labels.shape, test_labels.dtype)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  
    # Normalize data dimensions so that they are of approximately the same scale
    train_imgs = train_imgs / 255.
    test_imgs = test_imgs / 255.
    
    plt.figure(figsize=(10,10))
    for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(train_imgs[i], cmap=plt.cm.binary)
      plt.xlabel(class_names[train_labels[i]])
    plt.show()
#     print(validation_imgs.shape)
#     print(validation_labels.shape)
    
    np.savez('fashion_mnist_dataset.npz', 
    train_imgs=train_imgs, train_labels=train_labels, test_imgs=test_imgs, test_labels=test_labels)

def loadDataset():
    with np.load('fashion_mnist_dataset.npz') as npzfile:
        train_imgs = npzfile['train_imgs']
        train_labels = npzfile['train_labels']
        test_imgs = npzfile['test_imgs']
        test_labels = npzfile['test_labels']
    
    print('x_train : ', train_imgs.shape, train_imgs.dtype)
    print('y_train : ', train_labels.shape, train_labels.dtype)
    print('x_test : ', test_imgs.shape, test_imgs.dtype)
    print('y_test : ', test_labels.shape, test_labels.dtype)
    
    train_imgs, validation_imgs = np.split(train_imgs, 2)
    train_labels, validation_labels = np.split(train_labels, 2)

    model = keras.Sequential()
    epochs = 10
    
    # Must define the input shape in the first layer of the neural network
    model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Take a look at the model summary
    model.summary()
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(train_imgs, train_labels, batch_size=64, epochs=epochs, validation_data=(validation_imgs, validation_labels))
    
    score = model.evaluate(test_imgs, test_labels)
    print('\n', 'Test accuracy:', score[1])

def preprocessData():
    pass

if __name__ == '__main__':
    saveDataset()
    loadDataset()