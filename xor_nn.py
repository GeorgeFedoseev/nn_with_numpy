import numpy as np


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

NUM_EPOCHS = 50000

def loss(got, expected):
    return expected-got

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1 - sigmoid(x))

learning_rate = .1

input_size, hidden_size, output_size = 2, 3, 1

w_hidden = np.random.uniform(size=(input_size, hidden_size))
w_output = np.random.uniform(size=(hidden_size, output_size))

for epoch in range(NUM_EPOCHS):
    # iterate over data
    
    h = np.dot(X, w_hidden)
    act_hidden = sigmoid(h)
    output = np.dot(act_hidden, w_output)


    error = y - output
    
    if epoch % 5000 == 0:        
        print(f'loss {sum(error)}')

    # Backward        
    err_z = error * learning_rate
    # w_o += err_z * dz/dw_o = err_z * sigma(h)
    w_output += np.matmul(act_hidden.T, err_z)

    # err_h = err_z * dz/dh = err_z * dz/d(sigma(h)) * d(sigma(H))/d(h) = err_z * w_o * d(sigma(H))/d(h)
    err_h = np.matmul(err_z, w_output.T) * sigmoid_prime(h)
    w_hidden += np.matmul(X.T, err_h)
        






