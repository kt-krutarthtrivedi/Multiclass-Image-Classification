"""
| **@created on:** 02/01/22
| **@author:** Krutarth Trivedi | ktrivedi@wpi.edu
|
| **Description:**
|   Create a 2-layer Softmax Neural Network to classify images of the fashion items from the 
|   Fashion MNIST Dataset 
| **Tuned and Trained on Google Colab**
"""
import numpy as np
import matplotlib.pyplot as plt

# Default Hyperparameters
default_hyperparameters = {
    "epoch": 10,
    "mini_batch_size": 32,
    "reg_coeff": 0.1,
    "learning_rate": 1e-5,      
}

# % of Train Data to be considered as Valid
validation_set = 0.20

class SoftmaxRegression(object):
    def __init__(self):
        """
        Initialize weights and bias to None
        """
        self.weights = None
        self.bias = None

    def batch(self, iterable1, iterable2, n=1):
        """
        Function to batch the data
        :param iterable1: X
        :param iterable2: Y
        :param n: batches
        :return: batches of X and Y
        """
        l = len(iterable1)
        for ndx in range(0, l, n):
            yield iterable1[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)]

    def softmax_activation(self, ip):
        """
        Softmax Activation
        :param ip: Input
        :return: Probabilities
        """
        data = (np.exp(ip).T / np.sum(np.exp(ip), axis=1)).T
        return data

    def fit(self, X, Y, epoch=10, mini_batch_size=200, learning_rate=1e-3, reg_coeff=1e-3):
        """
        Fit the model
        :param X: Input
        :param Y: Label
        :param epoch: Iterations
        :param mini_batch_size: Batch Size
        :param learning_rate: Learning Rate
        :param reg_coeff: Regularization Coefficient
        """

        # Initialize weights with random uniform distribution
        self.weights = np.random.randn(784, 10) / np.sqrt(10 * 784)  # (784, 10)
        self.bias = np.random.randn(10) / np.sqrt(10)  # (10,)

        # Iterations
        for i in range(epoch):
            # Maintain Batch Loss
            batch_cost = []
            batch_acc = []
            # Iterate for every batch
            for x, y in self.batch(X, Y, mini_batch_size):
                # Get prediciton using equation: yhat = softmax( x * w ) + b
                yhat = self.predict(x)
                # Calculate loss
                loss = self.cost_fn(yhat, y)
                # Calculate Regularization Loss component
                reg_loss = (0.5 * reg_coeff * np.sum(self.weights * self.weights)) / mini_batch_size
                # Calculate total loss
                total_loss = loss + reg_loss
                batch_cost.append((total_loss, loss, reg_loss))
                batch_acc.append(self.accuracy(y, yhat))

                # Calculate gradient for regularized weights
                m_gradient = reg_coeff * ((-1 / mini_batch_size) * np.dot(x.T, (y - yhat)))

                # Calculate gradient for bias
                b_gradient = (-1 / mini_batch_size) * np.sum((y - yhat),axis=0)

                # Apply gradients for weights
                self.weights = self.weights - (learning_rate * m_gradient)
                # Apply gradients for bias
                self.bias = self.bias - (learning_rate * b_gradient)
            # Mean loss of all the batches
            epoch_loss = np.mean(batch_cost, axis=0)
            epoch_acc = np.mean(batch_acc, axis=0)
            print(f'Epoch {i} | Total Loss: {epoch_loss[0]} | Loss: {epoch_loss[1]} | RL: {epoch_loss[2]} | Acc: {epoch_acc}')

    def accuracy(self, y, y_hat):
        """
        Calculate accuracy
        :param y: Labels
        :param y_hat: Predictions
        :return: Softmax Accuracy
        """
        return np.mean(np.equal(np.argmax(y, axis = 1), np.argmax(y_hat, axis = 1)).astype(np.float32))
    
    def tune(self, X, Y, XVAL, YVAL, hyperparameter_tryouts: dict):
        """
        Tune the model to find best hyperparameters
        :param X: Input
        :param Y: Label
        :param XVAL: Validation Input
        :param YVAL: Validation Label
        :param hyperparameter_tryouts: Hyperparameter ranges to tryout
        :return: Best Hyperparmeters
        """
        # Initial best hyperparmeters = default hyperparameters
        best_hyper_params = default_hyperparameters.copy()
        # Keep track of loss history for each hyperparameter
        param_loss_history = {k: float('inf') for k, v in hyperparameter_tryouts.items()}
        for param, param_range in hyperparameter_tryouts.items():
            print(f'Tuning for parameter: {param} . . .')
            for p_val in param_range:
                print(f"Trying out {param} val: {p_val} . . .")
                # Set parameter
                default_hyperparameters[param] = p_val
                # Fit using given hyperparameter
                self.fit(X, Y, **default_hyperparameters)
                # Calculate validation loss
                validation_loss, validation_acc = self.evaluate(XVAL, YVAL)
                print(f'Validation Loss: {validation_loss} | Validation Accuracy: {validation_acc}')
                # Replace hyperparameter value based on condition
                if param_loss_history[param] > validation_loss:
                    param_loss_history[param] = validation_loss
                    best_hyper_params[param] = p_val
            print(f'Best Value for {param}: {best_hyper_params[param]}')
        return best_hyper_params

    def predict(self, x):
        """
        Given an instance x predict the label
        :param x: Input
        """
        return self.softmax_activation(np.matmul(x, self.weights) + self.bias)

    def evaluate(self, x, y):
        """
        Evaluate Cost on valid/test data
        :param x: Input
        :param y: Label
        """
        y_hat = self.predict(x)
        loss = self.cost_fn(y_hat, y)
        acc = self.accuracy(y, y_hat)
        return loss, acc

    def cost_fn(self, y_hat, y):
        """
        Cross Entropy cost function
        :param y_hat: Predicted
        :param y: Label
        :return: float value, mse loss
        """
        m = y.shape[1]
        return -(1. / m) * np.sum(np.multiply(y, np.log(y_hat)))

def preprosessing(y):
    y = y.reshape(y.shape[0], 1)  

    for i in range(len(y)):
        read = y[i]
        row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        row[read] = 1

        if i == 0:
            label_matrix = row
        else:
            label_matrix = np.vstack([label_matrix, row])

    return label_matrix    

def Fashion_MNIST_regressor():
    # Load Training Data
    xtr = np.load("fashion_mnist_train_images.npy")  # (60000, 784)
    ytr = np.load("fashion_mnist_train_labels.npy")  # (60000,)

    ytr = preprosessing(ytr)    #(60000, 10)
 
    # #random check with training examples
    # np.random.seed(0)
    # indices = list(np.random.randint(xtr.shape[0],size=10))
    # for i in range(10):
    #     plt.subplot(1,10,i+1)
    #     plt.imshow(xtr[indices[i]].reshape(28,28), cmap='gray', interpolation='none')
    #     plt.title("Class {}". format(ytr[i]))
    #     plt.tight_layout()
    # plt.show()

    # Create Valid Split
    val_index = xtr.shape[0] - int(xtr.shape[0] * validation_set)
    xtr, ytr, xval, yval = xtr[:val_index], ytr[:val_index], xtr[val_index:], ytr[val_index:]
 
    #Load Test Data
    xte = np.load( "fashion_mnist_test_images.npy")
    yte = np.load("fashion_mnist_test_labels.npy")

    yte = preprosessing(yte)        #(10000, 10)
 
    # Initialize the model
    model = SoftmaxRegression()

    hyperparameter_tryouts = {
        "epoch": [20, 50, 100, 150],
        "mini_batch_size": [32, 64, 128, 256],
        "reg_coeff": [1e-1, 1e-2, 1e-3, 1e-4],
        "learning_rate": [1e-4, 1e-5, 1e-6, 1e-7]
    }

    best_hyperparameters = model.tune(X=xtr, Y=ytr, XVAL=xval, YVAL=yval, hyperparameter_tryouts=hyperparameter_tryouts)

    # # Train the model for best hyper parameters
    print(f'Training model for best hyperparameters: {best_hyperparameters} . . .')

    model.fit(X=xtr, Y=ytr, **best_hyperparameters)

    print('************************************************************************')
    print()
    # #Printing the best hyperparameters
    print(f'Best hyperparameters: {best_hyperparameters}')
    
    # # Evaluate the model
    print('Model loss score on valid data:')
    print(model.evaluate(xval, yval))
    print('Model Loss Score on Test Data:')
    print(model.evaluate(xte, yte))

    print('************************************************************************')
    print()


if __name__ == '__main__':
    print('*********************** SoftMax Regression - Start ***********************')
    Fashion_MNIST_regressor()
    print('*********************** SoftMax Regression - End ***********************')