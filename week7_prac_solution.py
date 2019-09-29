# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 09:10:58 2018

@author: frederic

Solutions to prac sheet of W07

Modified: Yue Xu, Aug. 2019

"""


#------------------------------------------------------------------------------
def example_1():
    T = (9,3,7,2)
    it = iter(T)  # iterator
    print(next(it))
    print(next(it))
    print(next(it))
    print(next(it))
    print(next(it)) #  StopIteration  triggered
    
#------------------------------------------------------------------------------
def example_2():    
    
    iterable = (9,3,7,2)
    # create an iterator object from that iterable
    it = iter(iterable)
    
    # infinite loop
    while True:
        try:
            # get the next item
            element = next(it)
            # do something with element
            print(element)
        except StopIteration:
            # if StopIteration is raised, break from loop
            break    

#------------------------------------------------------------------------------
def example_3():
    # A simple generator function
    def my_gen():
        n = 1
        print('This is printed first')
        # Generator function contains yield statements
        yield n
    
        n += 1
        print('This is printed second')
        yield n
    
        n += 1
        print('This is printed at last')
        yield n
    
    # Using for loop
    for item in my_gen():
        print(item)


#------------------------------------------------------------------------------            
def ex_1():
    class PowTwo:
        """Class to implement an iterator
        of powers of two"""
    
        def __init__(self, max_val = 0):
            '''
            Stop iterating when the value returned is larger
            than 'max_val'
            '''
            self.max_val = max_val
                
    #-----------Your code to define __iter__(self) and __next__(self) -------------
            
        def __iter__(self):
            pass
            self.n = 0 # initialization
            return self
    
        def __next__(self):
            pass
            result = 2 ** self.n
            self.n += 1
            if result <= self.max_val:
                return result
            else:
                raise StopIteration
    #----------------------------
    
    # Create an object of PowTwo()
    pt = PowTwo(500)
    # it should print out 1, 2, 4, 8, ...., 128, 256
    for v in pt:
        print(v)

  
        
#------------------------------------------------------------------------------
def ex_2():
    #-----------Your code to define method PowTwoGen(max_val = 0) -------------
    def PowTwoGen(max_val = 0):
        pass
        result = 1 # 2**0
        while  result < max_val:
            yield result
            result *= 2

            
    #----------------------------
    # Create an object of PowTwo()
    pt = PowTwoGen(500)
    # it should print out 1, 2, 4, 8, ...., 128, 256
    for v in pt:
        print(v)
        

#------------------------------------------------------------------------------

def ex_3():
    from sklearn import datasets
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense 
    from keras.utils import to_categorical
    from sklearn import model_selection
    import numpy as np

    # The digits dataset
    digits = datasets.load_digits()

    # Number of samples
    n_samples = digits.data.shape[0]
    
    # Generate a random list of numbers from 0 to num_samples - 1 using np.random.permutation()
    random_per = np.random.permutation(n_samples)
    # Shuffle the dataset, X and Y are the shuffled dataset and target list
    X, Y = digits.data[random_per], digits.target[random_per] 

    # Split dataset into training and testing datasets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                    X, Y, test_size=0.2)

    # one-hot encode the labels - we have 10 output classes
    # we want to convert the label integer to a binary vector
    # e.g., 3 to [0 0 0 1 0 0 0 0 0 0], 5 to [0 0 0 0 0 1 0 0 0 0] 
    # this is done by using the to_categorical method
    num_classes = 10
    train_labels_cat = to_categorical(y_train, num_classes)
    test_labels_cat = to_categorical(y_test, num_classes)

    #-----------Your code goes here----------------------------------------------
      
    batch_size = 32
    Hidden_nodes = 20
    num_epochs = 30

    # Create a Sequential object 
    model = Sequential()

    # Add a hidden layer of 20 nodes
    model.add(Dense(Hidden_nodes,
                input_shape=(X_train.shape[1],),
                activation = 'relu')
              )
     
    # Add another hidden layer of 20 nodes
    model.add(Dense(Hidden_nodes, activation='relu'))

    # Add the output layer of 10 node
    model.add(Dense(num_classes, activation='softmax'))

    # Configure the model for training
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, train_labels_cat, epochs=num_epochs, batch_size=batch_size,
          validation_data=(X_test, test_labels_cat)) 

    # Evaluate the prediction result by using the test dataset
    evaluation_result = model.evaluate(X_test, test_labels_cat)
    print('\nTest loss:', evaluation_result[0])
    print('Test mae:', evaluation_result[1])


#------------------------------------------------------------------------------    
if __name__ == '__main__':
#     example_1()
#     example_2()
#     example_3()
#     ex_1()
#     ex_2()
     ex_3()
 
    
