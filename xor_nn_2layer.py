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

# network layers:
# X -> h1 -> h2 -> z 
w_h1 = np.random.uniform(size=(input_size, hidden_size))
w_h2 = np.random.uniform(size=(hidden_size, hidden_size))
w_z = np.random.uniform(size=(hidden_size, output_size))

def forward_pass(X, epoch=1):        
    # if epoch % 5000 == 0:            
        # print(w_h1)

    h1 = np.dot(X, w_h1)
    sigma_h1 = sigmoid(h1)
    h2 = np.dot(sigma_h1, w_h2)
    sigma_h2 = sigmoid(h2)
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
    err_h2 = np.matmul(err_z, w_z.T) * sigmoid_prime(h2)
    w_h2 += np.matmul(sigma_h1.T, err_h2)

    # update h1 layer weights
    err_h1 = np.matmul(err_h2, w_h2.T) * sigmoid_prime(h1)
    w_h1 += np.matmul(X.T, err_h1)
        
print("Test:")
print("Input | Result | Expected")
h1, sigma_h1, h2, sigma_h2, output = forward_pass(X)
for X, output, y in zip(X, output, y):
    print(X, np.round(output), y)







