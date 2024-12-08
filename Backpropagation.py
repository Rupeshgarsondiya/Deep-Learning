'''
Name   : Rupesh Garsondiya
github : @Rupeshgarsondiya
Topic  : Hands-on-implementation of neural network and backpropagation Algoritham Deep-Learning(DL)
'''

import pandas as pd
import numpy as np



class MY_ANN :
    def __init__(self):
        pass

    '''This function assigns the weights and biases of a neural network.'''
    def Initail_weight(self): 
        L = int(input('Enter number of layer : '))# Take input for the number of layers in a neural network from the user.
        self.Architecture = [] # Architecture is list of number of neurons in each layer
        self.parameter = {} # parameter is dictionary of weights and biases
        for i in range(L):
            node = int(input("Enter Layer node : "))
            self.Architecture.append(node)

        np.random.seed(len(self.Architecture))
        l = len(self.Architecture)
        for l in range(1,l):
            self.parameter['W' + str(l)] = np.ones((self.Architecture[l-1 ], self.Architecture[l]))*0.1
            self.parameter['b' + str(l)] = np.zeros((1, self.Architecture[l]))
        
        print(len(self.parameter))

        return self.parameter
  
    def fit(self,x_train,y_train,epochs=10):
        '''This function trains the neural network using backpropagation algorithm'''
        self.Initail_weight()
        print('Arcitecture',self.Architecture)
        print('weight : ',self.parameter.get('W1'))
        print(' bais : ',self.parameter.get('b1'))
        print('bais : ',self.parameter.get('b2'))

    def predict(slef,x_test):
        pass

df = pd.read_csv('/home/rupeshgarsondiya/workstation/Data-Set/Student_CGPA_Pakage.csv')

from sklearn.model_selection import train_test_split

x = df[['CGPA','IQ (*10)']]
y = df['Pakage']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = MY_ANN()
model.fit(x_train,y_train,10)
model.predict(x_test)
