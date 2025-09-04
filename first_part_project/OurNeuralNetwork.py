import numpy as np
from ActivationFunctions import Activation
from numba import njit


class OurNeuralNetwork:
    def __init__(self, input_layer_size: int, hidden_layers: np.ndarray, output_layer_size: int,
                 learning_rate: float, active_function: list[Activation], verbose: bool = False):
        if len(active_function) != len(hidden_layers) + 1:
            print("Number of active function must be equal to the number of weights")
        self.input_layer_size = input_layer_size
        self.hidden_layers = hidden_layers
        self.num_hidden_layers = len(hidden_layers)
        self.output_layer_size = output_layer_size
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate
        self.active_function = active_function
        self.verbose = verbose

    def init_weights(self):
        if self.verbose:
            print("Number of neurons in input layer is: ", str(self.input_layer_size))
        previous_layer_size = self.input_layer_size

        # hidden layers weights
        for i in range(self.num_hidden_layers):
            W = (np.random.random((self.hidden_layers[i],
                                  previous_layer_size))  - 0.5) / 1e3 # number of neurons in HL x number of inputs of each neuron
            b = (np.random.random((self.hidden_layers[i], 1)) - 0.5) / 1e3
            self.weights.append(W)
            self.biases.append(b)
            previous_layer_size = self.hidden_layers[i]

        # weights for output layer
        W = (np.random.random((self.output_layer_size,
                              previous_layer_size))  - 0.5) / 1e3 # number of neurons in HL x number of inputs of each neuron
        b = (np.random.random((self.output_layer_size, 1)) - 0.5) / 1e3
        self.weights.append(W)
        self.biases.append(b)

        if self.verbose:
            for i, a in enumerate(self.weights):
                print(f"\t->W_{i}", a.shape)
                print(f"\t->b_{i}", self.biases[i].shape)

    def forward(self, X: np.ndarray):
        """
            Forward propagation of the NN
            - X - input to the first layer
        """
        if self.verbose:
            print("\t\t\tX:", X.shape)
        if X.shape[1] == self.input_layer_size and X.shape[0] != self.input_layer_size:
            Z = [X.reshape((X.shape[1], X.shape[0]))]  # before activation, reshape for proper data shape
            A = [X.reshape((X.shape[1], X.shape[0]))]  # after activation
        elif X.shape[0] == self.input_layer_size:
            Z = [X]
            A = [X]
        else:
            print(
                "Shape of the input is not correct. Number of input neurons is different than length of input vector.")
            return 1
        for i in range(self.num_hidden_layers + 1):  # + 1 because of output layer
            Zi = np.dot(self.weights[i], A[i]) + self.biases[i]
            Ai = self.active_function[i].apply(Zi)
            A.append(Ai)
            Z.append(Zi)
        if self.verbose:
            for i, a in enumerate(A):
                print(f"\t\t->A_{i}", a.shape)
                print(f"\t\t->Z_{i}", Z[i].shape)
        return Z[::-1], A[::-1]  # the first index will be the output of NN

    def backpropagation(self, Z, A, Y):
        """
            Backpropagation of the NN
            - Z - outputs of the forward propagation - not activated
            - A - outputs of the forward propagation - activated
            - Y - output labels!
        """
        if Y.shape[1] == self.output_layer_size and Y.shape[0] == A[-1].shape[1]:
            Y = Y.reshape((Y.shape[1], Y.shape[0]))  # reshape for proper data shape
        elif Y.shape[0] != self.output_layer_size or Y.shape[1] != A[-1].shape[1]:
            print(
                "Shape of the input is not correct. Number of input neurons is different than length of input vector.")
            return 1
        dW = []
        db = []
        M = Y.shape[0]
        dZi = Y
        for i in range(self.num_hidden_layers + 1):
            k = self.num_hidden_layers - i + 1
            if i == 0:
                dZi = A[self.num_hidden_layers + 1] - Y
            else:
                deriv = self.active_function[self.num_hidden_layers - i].derivative(Z[k])
                # start from the left side by the previous weights
                dZi = dZi.dot(self.W[k])
                # elementwise
                dZi = dZi * deriv

            dWi = 1.0 / M * dZi.T.dot(A[self.num_hidden_layers - i])
            dbi = np.mean(dZi, axis=0)[:, None]
            dW.append(dWi)
            db.append(dbi)
        if self.verbose:
            for i in range(self.num_hidden_layers + 1):
                print(f"\t\t->dW_{i}", dW[i].shape)
                print(f"\t\t->db_{i}", db[i].shape)
        return dW[::-1], db[::-1]

    def update(self, dW, db):
        '''
        Having the gradients - update the weights!
        '''

        for i in range(self.num_hidden_layers):
            self.weights[i] = self.weights[i] - self.learning_rate * dW[i]
            self.biases[i] = self.biases[i] - self.learning_rate * db[i]

        self.weights[-1] = self.weights[-1] - self.learning_rate * dW[-1]
        self.biases[-1] = self.biases[-1] - self.learning_rate * db[-1]

    def getAccuracy(self, yTrue, yPred):
        '''
        Check the accuracy of the prediction
        - yTrue - true labels put to the NN
        - yPred - predicted labels
        '''
        # return yTrue == yPred
        return np.argmax(yTrue, axis = 0) == np.argmax(yPred, axis = 0)
    def saveWeights(self):
        np.save('weights1.npy', self.weights[0])
        np.save('weights2.npy', self.weights[1])

        np.save('bias1.npy', self.biases[0])
        np.save('bias2.npy', self.biases[1])

    def gradientDescent(self, X, Y, iterations):
        '''
        Loop to update the NN parameters (with gradient descent)
        - X          - images
        - Y          - labels
        - iterations - number of interations to finish
        '''
        # initialize the weights
        self.init_weights()
        print("Y:", Y.shape)
        history = []
        # iterate
        for i in range(iterations):
            # calculate forward propagation
            Z, A = self.forward(X)
            # calculate backward propagation
            dW, db = self.backpropagation(Z, A, Y)
            # update the weights
            self.update(dW, db)

            # get the accuracy
            accuracy = self.getAccuracy(Y, A[-1])
            accuracy = np.mean(accuracy)
            history.append(np.mean(accuracy))

            # print accuracy
            if i % 50 == 0:
                print(self.weights[0])
                print(self.biases[0])
                print("W1")
                print(self.weights[1])
                print(self.biases[1])
                print('----------------------------')
                print(np.array(A[0])[:, 0])
                print(list(Y[:, 0]))
                print(f"Iteration [{i}/{iterations}]")
                print(f"Accuracy: {accuracy:.3f}")
                print('-------------------------')
        return history

    def train(self, X, Y, iterations):
        return self.gradientDescent(X, Y, iterations)

    def test(self, X, Y):
        A, Z = self.forward(X)
        acc = self.getAccuracy(Y, A[-1])
        return acc, np.mean(acc), A[-1]
