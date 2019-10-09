import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as k
import matplotlib.pyplot as plt
import random

# define the names of classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_classes = len(class_names)
# Set 1 is the images with labels for classes "top", "trouser", "pullover", "coat", "sandal", "andke boot"
labels1 = [0, 1, 2, 4, 5, 9]

# Set 2 is the images with labels for classes "dress", "sneaker", "bag", "shirt"
labels2 = [3, 6, 7, 8]

def Save_Dataset():
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
    
    ##### Check the type of each dataset #####
  # print('x_train : ', train_imgs.shape, train_imgs.dtype)
  # print('y_train : ', train_labels.shape, train_labels.dtype)
  # print('x_test : ', test_imgs.shape, test_imgs.dtype)
  # print('y_test : ', test_labels.shape, test_labels.dtype)
  

  ##### View the first 25  objects #####
  # plt.figure(figsize=(10,10))
  # for i in range(25):
  #   plt.subplot(5,5,i+1)
  #   plt.xticks([])
  #   plt.yticks([])
  #   plt.grid(False)
  #   plt.imshow(train_imgs[i], cmap=plt.cm.binary)
  #   plt.xlabel(class_names[train_labels[i]])
  # plt.show()
  # print(validation_imgs.shape)
  # print(validation_labels.shape)
  
  # Save datasets as a local file
  np.savez('fashion_mnist_dataset.npz', 
  train_imgs=train_imgs, train_labels=train_labels, test_imgs=test_imgs, test_labels=test_labels)

def Load_Dataset():
  '''
  This method is for building the a weight learning for Convolutional Neroun Network
  1. Load the file with datasets
  2. Divide training dataset into train and validation
  3. Pre-process the dataset
  4. Build a model adn input two datasets
  5. 
  '''
  # Load dataset from a local file 
  with np.load('fashion_mnist_dataset.npz') as npzfile:
      train_imgs = npzfile['train_imgs']
      train_labels = npzfile['train_labels']
      test_imgs = npzfile['test_imgs']
      test_labels = npzfile['test_labels']
  # Normalize data dimensions so that they are of approximately the same scale
  train_imgs = train_imgs.astype('float32')
  test_imgs = test_imgs.astype('float32')
  train_imgs /= 255.
  test_imgs /= 255.

  # the rows and columns of the images are at position 1 and 2
  img_rows, img_cols = train_imgs.shape[1:3]
  
  # Reshape the input arrays to 4D (batch_size, rows, columns, channels)
  train_imgs = train_imgs.reshape(train_imgs.shape[0], img_rows, img_cols, 1)
  test_imgs = test_imgs.reshape(test_imgs.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)
  batch_size = 128
  epochs = 10

  # convert class vectors to binary class matrices (aka "sparse coding" or "one hot")
  # train_labels = keras.utils.to_categorical(train_labels, num_classes)
  # test_labels = keras.utils.to_categorical(test_labels, num_classes)
  # print('Training Labels:', train_labels)
  # print('Testing Labels:', test_labels)
  
  # create training+test positive and negative pairs
  digit_indices = [np.where(train_labels == i)[0] for i in labels1]
  tr_pairs, tr_y = create_pairs(train_imgs, digit_indices, labels1)

  digit_indices = [np.where(test_labels == i)[0] for i in labels1]
  te_pairs, te_y = create_pairs(test_imgs, digit_indices, labels1)

  base_network = Create_Base_Network(input_shape)
  input_a = keras.Input(shape=input_shape)
  input_b = keras.Input(shape=input_shape)

  processed_a = base_network(input_a)
  processed_b = base_network(input_b)
  print(processed_a)
  distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  model = keras.Model([input_a, input_b], distance)
  model.summary()
  # Compile the model
  model.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=['accuracy'])
  # Train the model
  model.fit([tr_pairs[:, 0, :,:,:], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=epochs, validation_data=([te_pairs[:, 0,:,:,:], te_pairs[:, 1]], te_y))
  # model.fit([train_imgs[:, 0], train_imgs[:, 1]], train_labels, batch_size=batch_size, epochs=epochs, validation_data=([test_imgs[:, 0], test_imgs[:, 1]], test_labels))
  score = model.evaluate(test_imgs, test_labels)
  print('\n', 'Test accuracy:', score[1])

def euclidean_distance(vects):
    x, y = vects
    sum_square = k.sum(k.square(x - y), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_square, k.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = k.square(y_pred)
    margin_square = k.square(k.maximum(margin - y_pred, 0))
    return k.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_pairs(x, digit_indices, dataset):
  '''Positive and negative pair creation.
  Alternates between positive and negative pairs.
  '''
  pairs = []
  labels = []
  n = min([len(digit_indices[d]) for d in range(len(dataset))]) - 1

  for d in range(len(dataset)):
      for i in range(n):
          z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
          pairs += [[x[z1], x[z2]]]
          inc = random.randrange(1, len(dataset))
          dn = (d + inc) % len(dataset)
          z1, z2 = digit_indices[d][i], digit_indices[dn][i]
          pairs += [[x[z1], x[z2]]]
          labels += [0, 1]
  return np.array(pairs), np.array(labels)

def Create_Base_Network(input_shape):
  '''
    Create a sharable base network for two input datasets
  '''
  # # Express output as probability of image belonging to a particular class
  # model.add(keras.layers.Dense(num_classes, activation='softmax'))
  # #The output of the CNN is a number of propability

  ip = keras.Input(shape=input_shape)
  # The 'relu' parameter is used to replace all negative values by zero, relu is abbreviated as Rectified Linear Unit
  layer1 = keras.layers.Conv2D(64, kernel_size=2, padding='same', activation='relu')(ip)
  layer1 = keras.layers.MaxPooling2D(pool_size=2)(layer1)
  layer1 = keras.layers.Dropout(0.3)(layer1)

  layer2 = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(layer1)
  layer2 = keras.layers.MaxPooling2D(pool_size=2)(layer2)
  layer2 = keras.layers.Dropout(0.5)(layer2)
  layer2 = keras.layers.Flatten()(layer2)
  # layer2 = keras.layers.Dense(64, activation='relu')
  return keras.Model(ip, layer2)

if __name__ == '__main__':
    Save_Dataset()
    Load_Dataset()