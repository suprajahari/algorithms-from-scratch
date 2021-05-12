import numpy as np
import matplotlib.pyplot as plt
class Neural_net():
    
    # we need,
        # X, y, no_of_layers,
        # no_of_nodes, weight and bias
        # learning_rate, iterations
        # loss
        # sample size
        
    def __init__(self, layers = [13,8,1], learning_rate = 0.001, iterations = 100):

        # we took 2 layer network
            # input_layer - 13 nodes
            # hidden_layer1 - 8 nodes
            # output_layer - 1 node  ( consider we are using binary classification problem)
            # if we are using multiclass classification we can use multiple output layer

        self.params = {}
        self.X = None
        self.y = None
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        

    # now we are going to assign weight and bias to each layers
    # we are going to store those weight and bias in params dictionary
    # we have two layers and we are using random normal distribution

    def init_weights(self):

        np.random.randn.seed(1)
        self.params['w1'] = np.random.randn(self.layers[0],self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1])
        self.params['w2'] = np.random.randn(self.layers[1],self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2])

        '''
         we have two layers so we are going to add two active function
         (ex: 'relu', 'sigmoid', 'maxout', 'tanh', 'leaky relu')
        '''

    def relu(self, z):
        
        '''
        relu activation function is to performs a threshold operation to each input
        element where values less than zero are set to zero.
        '''

        return np.maximum(0,z)

    def sigmoid(self, z):

        '''
        the sigmoid activation function takes in real number in any range and
        squashes it to a real-valued output between 0 and 1.
        '''

        return 1.0/(1.0 + np.exp(-z))

    ''' Loss function to reduce the error'''

    def entropy_loss(self, y, predicted):

        m = length(y)
        loss = (-1/m) * (np.sum(np.multiply(np.log(predicted),y) + np.multiply((1-y),np.log(predicted))))
        return loss

    
    ''' we are going to perform forward propogation

        Z1 = (w1 * X) + b1
        A1 = Relu(Z1) # because we are using relu activation between input and first hidden layer
        Z2 = (w2 * A1) + b2
        A2 = sigmoid(Z2) # because we are using sigmoid activation between hidden layer and output layer
        loss = (A2 , y)
    '''

    def forward_propagation(self):

        z1 = self.X.dot(self.params['w1']) + self.params['b1']
        A1 = self.relu(z1)
        z2 = A1.dot(self.params['w2']) + self.params['b2']
        predicted = self.sigmoid(z2)
        loss = self.entropy_loss(self.y, predicted)

        self.params['z1'] = z1
        self.params['z2'] = z2
        self.params['A1'] = A1

        return predicted,loss

    '''
        we are going to do back propagation algorithm
    '''

    def back_propagation(self,predicted):

        def drelu(X):

            X[X<=0] = 0
            X[X>0] = 0

            return X 
        
        loss_predicted = np.divide(-self.y,predicted) + np.divide((1-self.y),(1-predicted))
        loss_sigmoid = predicted * (1 - predicted)
        loss_z2 = loss_predicted * loss_sigmoid

        loss_A1 = loss_z2.dot(self.params['w2'].T)
        loss_w2 = self.params['A1'].T.dot(loss_z2)
        loss_b2 = np.sum(loss_z2, axis = 0)

        loss_z1 = loss_A1 * drelu(self.params['z1'])
        loss_w1 = self.X.T.dot(loss_z1)
        loss_b1 = np.sum(loss_z1, axis = 0)

    '''update weight and bias'''

        self.params['w1'] = self.params['w1'] - self.learning_rate * loss_w1
        self.params['w2'] = self.params['w2'] - self.learning_rate * loss_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * loss_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * loss_b2

    ''' fit the parameters to the algorithm'''

    def fit(self,X,y):

        '''
        trains neural network with given data and labels
        '''
        self.X = x
        self.y = y
        self.init_weights()

        for i in range(self.iterations):
            predicted,loss = self.forward_propagation()
            self.back_propagation()
            self.loss.append(loss)

    def predict(self,X):

        ''' for prediction we have to just use forward propagation'''

        z1 = self.X.dot(self.params['w1']) + self.params['b1']
        A1 = self.relu(z1)
        z2 = A1.dot(self.params['w2']) + self.params['b2']
        predicted = self.sigmoid(z2)
        return np.round(predicted)


    def accuracy(self, y, predicted):

        acc = (y - predicted)/len(y)
        return acc*100

    def plot_loss(self):

        plt.plot(self.loss)
        plt.xlabel('iteration')
        plt.ylabel('log loss')
        plt.title('loss for training')
        plt.show()





        
        
    
