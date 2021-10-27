import numpy as np

def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return x * (x > 0)

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if x<=0:
        return 0
    else:
        return 1

# def train(x1, x2, y):

#         ### Forward propagation ###
#         input_values = np.matrix([[x1],[x2]]) # 2 by 1

#         W=np.matrix('1. 2.; 3. 4.; 5. 6.') #np.array([[1,2],[3,4],[5,6]]) #[w11,w21],[w12,w22],[w13,w33]
#         V=np.matrix('1. 2. 3.')
        
#         # Calculate the input and activation of the hidden layer
#         hidden_layer_weighted_input=W*input_values

#         vec_rectified_linear_unit=np.vectorize(rectified_linear_unit) 
#         hidden_layer_activation = vec_rectified_linear_unit(hidden_layer_weighted_input)   #(3 by 1 matrix)

#         output =  np.sum(np.multiply(V.T,hidden_layer_activation))

#         activated_output = output_layer_activation(output)

        

#         # ### Backpropagation ###

#         # # Compute gradients
#         cost=0.5*(y-output)**2
#         output_layer_error = -(y-output)
#         output_derivative_vec = np.vectorize(output_layer_activation_derivative)    # Vectorize derivative of output activation
#         hidden_layer_error = np.multiply(output_derivative_vec(activated_output),self.hidden_to_output_weights.transpose())*output_layer_error #(3 by 1 matrix)
        
#         ReLU_derivative_vec = np.vectorize(rectified_linear_unit_derivative) # Vectorize ReLU derivative
#         bias_gradients = np.multiply(hidden_layer_error, ReLU_derivative_vec(hidden_layer_weighted_input)) #dC/db

#         hidden_to_output_weight_gradients = np.multiply(hidden_layer_activation, output_layer_error).transpose() #dC/dV
#         input_to_hidden_weight_gradients = bias_gradients.dot(input_values.transpose()) #dC/dW

#         # Use gradients to adjust weights and biases using gradient descent
#         self.biases = self.biases - self.learning_rate*bias_gradients
#         self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate*input_to_hidden_weight_gradients
#         self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate*hidden_to_output_weight_gradients





#         # bias_gradients = # TODO
#         # hidden_to_output_weight_gradients = # TODO
#         # input_to_hidden_weight_gradients = # TODO

#         # # Use gradients to adjust weights and biases using gradient descent
#         # self.biases = # TODO
#         # self.input_to_hidden_weights = # TODO
#         # self.hidden_to_output_weights = # TODO

# print(train(1,2,50))

def train( x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        W=np.matrix('1. 2.; 3. 4.; 5. 6.')
        V=np.matrix('1. 1. 1.')
        B=np.matrix('0.; 0.; 0.')
        learn_rate=.001

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.dot(W,input_values) + B #(3 by 1 matrix)
        vec_rectified_linear_unit = np.vectorize(rectified_linear_unit)  # Vectorize ReLU function
        hidden_layer_activation = vec_rectified_linear_unit(hidden_layer_weighted_input) #(3 by 1 matrix)

        output = np.dot(V,hidden_layer_activation)
        activated_output = output_layer_activation(output)

        ### Backpropagation ###

        # Compute gradients
        output_layer_error = -(y - activated_output)
        
        vec_output_layer_activation_derivative = np.vectorize(output_layer_activation_derivative)
        hidden_layer_error = np.multiply(vec_output_layer_activation_derivative(activated_output),V.transpose())*output_layer_error #(3 by 1 matrix)
        
        vec_rectified_linear_unit_derivative = np.vectorize(rectified_linear_unit_derivative) # Vectorize ReLU derivative
        bias_gradients = np.multiply(hidden_layer_error, vec_rectified_linear_unit_derivative(hidden_layer_weighted_input))

        hidden_to_output_weight_gradients = np.multiply(hidden_layer_activation, output_layer_error).transpose()
        input_to_hidden_weight_gradients = bias_gradients.dot(input_values.transpose())

        # Use gradients to adjust weights and biases using gradient descent
        B = B - learn_rate*bias_gradients
        W = W - learn_rate*input_to_hidden_weight_gradients
        V = V - learn_rate*hidden_to_output_weight_gradients
        return hidden_layer_error

print(train( 2, 1, 10))