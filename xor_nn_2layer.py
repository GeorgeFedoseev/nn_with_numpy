import numpy as np


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

NUM_EPOCHS = 50000

def loss(got, expected):
    return expected-got

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def relu(x):
    if np.any(np.isnan(x)):
        print("nan", x)
        exit()
    return x * (x > 0)

def relu_derivative(x):    
    return 1. * (x > 0)


leaky_relu_slope = 0.01
def leaky_relu(x):
    return np.where(x > 0, x, x * leaky_relu_slope)

def leaky_relu_derivative(x):    
    return np.where(x > 0, 1, leaky_relu_slope)


learning_rate = .1

input_size, hidden_size, output_size = 2, 3, 1

activation = sigmoid
activation_derivative = sigmoid_derivative

# network layers:
# X -> h1 -> h2 -> z 
w_h1 = np.random.uniform(size=(input_size, hidden_size))
w_h2 = np.random.uniform(size=(hidden_size, hidden_size))
w_z = np.random.uniform(size=(hidden_size, output_size))

def forward_pass(X, epoch=1): 

    h1 = np.dot(X, w_h1)
    
    sigma_h1 = activation(h1)
    h2 = np.dot(sigma_h1, w_h2)
    sigma_h2 = activation(h2)
    output = np.dot(sigma_h2, w_z)     

    return h1, sigma_h1, h2, sigma_h2, output

# iterate over data
for epoch in range(NUM_EPOCHS):

    # Forward
    h1, sigma_h1, h2, sigma_h2, output = forward_pass(X, epoch) 

    error = y - output
    
    if epoch % 5000 == 0:               
        print(f'loss: {sum(error)}')        

    # Backward

    # update output layer weights
    err_z = error * learning_rate    
    w_z += np.matmul(sigma_h2.T, err_z)

    # update h2 layer weights
    err_h2 = np.matmul(err_z, w_z.T) * activation_derivative(h2)
    w_h2 += np.matmul(sigma_h1.T, err_h2)

    # update h1 layer weights
    err_h1 = np.matmul(err_h2, w_h2.T) * activation_derivative(h1)
    w_h1 += np.matmul(X.T, err_h1)
        
print("Test:")
print("Input | Result | Expected")
h1, sigma_h1, h2, sigma_h2, output = forward_pass(X)
for X, output, y in zip(X, output, y):
    print(X, np.round(output), y)







