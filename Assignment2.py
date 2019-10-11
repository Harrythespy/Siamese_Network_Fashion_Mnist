import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as k, regularizers
import matplotlib.pyplot as plt
import random

# define the names of classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Set 1 is the images with labels for classes "top", "trouser", "pullover", "coat", "sandal", "andke boot"
labels1 = [0, 1, 2, 4, 5, 9]

# Set 2 is the images with labels for classes "dress", "sneaker", "bag", "shirt"
labels2 = [3, 6, 7, 8]

# Create new dataset in order to store different classes of images and labels
Set1_imgs, Set1_labels = [], []
Set2_imgs, Set2_labels = [], []

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
   
  # Classify both images and labels with different classes
  for data in range(0, len(imgs)):
    if labels[data] in labels1:
      Set1_imgs.append(imgs[data])
      Set1_labels.append(labels[data])
    else:
      Set2_imgs.append(imgs[data])
      Set2_labels.append(labels[data])           
  
  # Since the initial type of the arrays are python arrays
  # Convert into numpy arrays
  Set1_imgs = np.array(Set1_imgs)
  Set1_labels = np.array(Set1_labels)
  Set2_imgs = np.array(Set2_imgs)
  Set2_labels = np.array(Set2_labels)
  
  ##### Check the type of each dataset #####
#   print('x_train : ', Set1_imgs.shape, Set1_imgs.dtype)
#   print('y_train : ', Set1_labels.shape, Set1_labels.dtype)
#   print('x_test : ', Set2_imgs.shape, Set2_imgs.dtype)
#   print('y_test : ', Set2_labels.shape, Set2_labels.dtype)

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
  # np.savez('fashion_mnist_dataset.npz', 
  # train_imgs=train_imgs, train_labels=train_labels, test_imgs=test_imgs, test_labels=test_labels)

def Load_Dataset():
  '''
  This method is for building the a weight learning for Convolutional Neroun Network
  1. Load the file with datasets
  2. Divide training dataset into train and validation
  3. Pre-process the dataset
  4. Build a model adn input two datasets
  '''
  
  def euclidean_distance(vects):
    x, y = vects
    sum_square = k.sum(k.square(x - y), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_square, k.epsilon()))

  
  def eucl_dist_output_shape(shapes):
      shape1, shape2 = shapes
      return (shape1[0], 1)

  
  def contrastive_loss(y_true, y_pred):
      '''
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
      '''
      margin = 1
      square_pred = k.square(y_pred)
      margin_square = k.square(k.maximum(margin - y_pred, 0))
      return k.mean(y_true * square_pred + (1 - y_true) * margin_square)


  def create_pairs(x, digit_indices, dataset):
    '''
      Positive and negative pair creation.
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
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


  def Create_Base_Network(input_shape):
    '''
      Create a sharable base network for two input datasets
    '''
    # # Express output as probability of image belonging to a particular class

    ip = keras.Input(shape=input_shape)

    # The 'relu' parameter is used to replace all negative values by zero, relu is abbreviated as Rectified Linear Unit
    layer1 = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(ip)
    # layer1 = keras.layers.MaxPooling2D(pool_size=2)(layer1)
    # layer1 = keras.layers.Dropout(0.3)(layer1)
    layer2 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu')(layer1)
    layer2 = keras.layers.MaxPooling2D(pool_size=(3,3))(layer2)
    layer2 = keras.layers.Flatten()(layer2)
    output = keras.layers.Dense(128, activation='relu')(layer2)
    layer2 = keras.layers.Dropout(0.2)(layer2)
    # layer2 = keras.layers.Dense(64, activation='relu')
    return keras.Model(ip, output)

  def compute_accuracy(y_true, y_pred):
      '''
        Compute classification accuracy with a fixed threshold on distances.
      '''
      pred = y_pred.ravel() < 0.5
      return np.mean(pred == y_true)


  def accuracy(y_true, y_pred):
      '''
        Compute classification accuracy with a fixed threshold on distances.
      '''
      return k.mean(k.equal(y_true, k.cast(y_pred < 0.5, y_true.dtype)))

  # Download the dataset from keras.fashion  
  # give names for the downloaded data
  (train_imgs, train_labels), (test_imgs, test_labels) = keras.datasets.fashion_mnist.load_data()

  #put training data and testing data together and them separate them as 80-20 with given condition
  imgs = np.concatenate((train_imgs, test_imgs))
  labels = np.concatenate((train_labels, test_labels))
   
  # Create new dataset in order to store different classes of images and labels
  Set1_imgs, Set1_labels = [], []
  Set2_imgs, Set2_labels = [], []
    
  # Classify both images and labels with different classes
  for data in range(0, len(imgs)):
    if labels[data] in labels1:
      Set1_imgs.append(imgs[data])
      Set1_labels.append(labels[data])
    else:
      Set2_imgs.append(imgs[data])
      Set2_labels.append(labels[data])           
  
  # Since the initial type of the arrays are python arrays
  # Convert into numpy arrays
  Set1_imgs = np.array(Set1_imgs)
  Set1_labels = np.array(Set1_labels)
  Set2_imgs = np.array(Set2_imgs)
  Set2_labels = np.array(Set2_labels)
  
  # Distribute training and testing datasets with 80/20 percent
  train_imgs, test_imgs, train_labels, test_labels = train_test_split(Set1_imgs, Set1_labels, test_size=0.2, random_state=87)

  # # Load dataset from a local file 
  # with np.load('fashion_mnist_dataset.npz') as npzfile:
  #     train_imgs = npzfile['train_imgs']
  #     train_labels = npzfile['train_labels']
  #     test_imgs = npzfile['test_imgs']
  #     test_labels = npzfile['test_labels']
  
  # Normalize data dimensions so that they are of approximately the same scale
  train_imgs = train_imgs.astype('float32')
  test_imgs = test_imgs.astype('float32')
  train_imgs /= 255.
  test_imgs /= 255.

  # the rows and columns of the images are at position 1 and 2
  img_rows, img_cols = train_imgs.shape[1:]
  input_shape = (img_rows, img_cols, 1)
  batch_size = 256
  epochs = 12
  
  # create positive and negative pairs of training and testing datasets
  digit_indices = [np.where(train_labels == i)[0] for i in labels1]
  tr_pairs, tr_y = create_pairs(train_imgs, digit_indices, labels1)

  digit_indices = [np.where(test_labels == i)[0] for i in labels1]
  te_pairs, te_y = create_pairs(test_imgs, digit_indices, labels1)

  # Testing with pairs from the set of images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"] union ["dress", "sneaker", "bag", "shirt"]
  test2_imgs = np.concatenate((test_imgs, Set2_imgs))
  test2_labels = np.concatenate((test_labels, Set2_labels))
  digit_indices = [np.where(test2_labels == i)[0] for i in (labels1 + labels2)]
  test2_pairs, test2_y = create_pairs(test2_imgs, digit_indices, (labels1 + labels2))
  
  # Create positive and negative pairs of test 3
  digit_indices = [np.where(Set2_labels == i)[0] for i in labels2]
  test3_pairs, test3_y = create_pairs(Set2_imgs, digit_indices, labels2)
  
  # Reshape the input arrays to 4D (batch_size, rows, columns, channels)
  tr_pairs = tr_pairs.reshape(tr_pairs.shape[0], 2, img_rows, img_cols, 1)
  te_pairs = te_pairs.reshape(te_pairs.shape[0], 2, img_rows, img_cols, 1)
  
  test2_pairs = test2_pairs.reshape(test2_pairs.shape[0], 2, img_rows, img_cols, 1)
  test3_pairs = test3_pairs.reshape(test3_pairs.shape[0], 2, img_rows, img_cols, 1)
  # Display the pairs of each class
  # fig, ax = plt.subplots(nrows=10, ncols=4,figsize=(40, 40))
  # idx = 0
  # for row in range(10):
  #     idx = random.randrange(0,len(tr_pairs),2)
  #     ax[row,0].imshow(tr_pairs[idx][0],cmap = 'gray')
  #     ax[row,1].imshow(tr_pairs[idx][1],cmap = 'gray')
  #     idx+=1
  #     ax[row,2].imshow(tr_pairs[idx][0],cmap = 'gray')
  #     ax[row,3].imshow(tr_pairs[idx][1],cmap = 'gray')
  # plt.show()
  # Chat Conversation End Type a message...

  # Implement the CNN with both inputs
  base_network = Create_Base_Network(input_shape)
  base_network.summary()
  input_a = keras.Input(shape=input_shape)
  input_b = keras.Input(shape=input_shape)

  processed_a = base_network(input_a)
  processed_b = base_network(input_b)
  
  distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
  model = keras.Model([input_a, input_b], distance)
  
  # Display the structure of the CNN
  model.summary()

  # Compile the model
  model.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[accuracy])

  # Distribute training and testing datasets with 80/20 percent
  train_pairs, valid_pairs, train_labels, valid_labels = train_test_split(tr_pairs, tr_y, train_size=0.8, random_state=87)
  
  # Train the model
  history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels, batch_size=batch_size, epochs=epochs, validation_data=([valid_pairs[:, 0], valid_pairs[:, 1]], valid_labels))
  
  # Plot training & validation accuracy values
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()
  
  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()
  
  y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
  tr_acc = compute_accuracy(tr_y, y_pred)
  y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
  te_acc = compute_accuracy(te_y, y_pred)
  
  # Testing with pairs from the set of images with labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"]
  print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
  print('* Accuracy on test 1 set: %0.2f%%' % (100 * te_acc))

  ##### Evaluate the model of test 2 #####
  test2_pred = model.predict([test2_pairs[:, 0], test2_pairs[:, 1]])
  test2_acc = compute_accuracy(test2_y, test2_pred)
  print('* Accuracy on test 2 set: %0.2f%%' % (100 * test2_acc))

  ##### Evaluate the model of test 3 #####
  test3_pred = model.predict([test3_pairs[:, 0], test3_pairs[:, 1]])
  test3_acc = compute_accuracy(test3_y, test3_pred)
  print('* Accuracy on test 3 set: %0.2f%%' % (100 * test3_acc))
  
if __name__ == '__main__':
#     Save_Dataset()
    Load_Dataset()