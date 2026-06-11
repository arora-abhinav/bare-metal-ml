import json
import math
import random
from abc import ABC, abstractmethod
from enum import Enum
from bare_metal_ml import linalg as custom_math
from bare_metal_ml.autograd import Matrix


#Base class for all activation functions. Users can inherit from this to define custom activations.
#Each subclass must implement forward() and derivative() as element-wise operations on scalars.
#The framework automatically wraps them in the computation graph via apply_activation().
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

#Predefined activation functions — users can instantiate these directly or inherit and override
class ReLU(ActivationFunction):
    def forward(self, x):
        return x if x > 0 else 0.0

    def derivative(self, x):
        return 1.0 if x > 0 else 0.0

class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(self, x):
        s = self.forward(x)
        return s * (1.0 - s)

class Tanh(ActivationFunction):
    def forward(self, x):
        return math.tanh(x)

    def derivative(self, x):
        return 1.0 - math.tanh(x) ** 2

#Convenience enum for selecting predefined activations without constructing them manually.
#Pass an ActivationFunction instance to override this entirely.
class FunctionType(Enum):
    RELU = 1
    SIGMOID = 2
    TANH = 3

#Wraps any ActivationFunction into a Matrix autograd node.
#forward() is applied element-wise to build the output matrix.
#back() uses derivative() element-wise and multiplies by the upstream gradient.
def apply_activation(z_node, activation_func):
    rows = len(z_node.matrix)
    cols = len(z_node.matrix[0])
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            result[r][c] = activation_func.forward(z_node.matrix[r][c])

    res = Matrix(result, {z_node}, "activation")
    res.gradient = [[0] * cols for _ in range(rows)]

    def back():
        #Apply the scalar derivative element-wise, then multiply by the upstream gradient
        deriv = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                deriv[r][c] = activation_func.derivative(z_node.matrix[r][c])
        deriv_times_grad = custom_math.element_wise_multiplication(deriv, res.gradient)
        z_node.gradient = custom_math.matrix_addition_and_sub(z_node.gradient, deriv_times_grad, "add")

    res.back = back
    return res


class LossFunctions:
    def __init__(self):
        pass

    def cross_entropy_loss(self, y_hat, y):
        clipped = self.regularise(y_hat)
        for i in range(len(clipped)):
            clipped[i] = math.log(clipped[i]) * y[i]
        return -sum(clipped)

    def regularise(self, vector, epsilon=1e-7):
        return [max(epsilon, min(1.0 - epsilon, v)) for v in vector]


class Layer:
    def __init__(self, neuron_num, input_size, activation=None, function_type=None):
        #Resolve activation: custom instance takes priority, otherwise use the predefined enum
        if activation is not None:
            self.activation = activation
        elif function_type is not None:
            if function_type == FunctionType.RELU:
                self.activation = ReLU()
            elif function_type == FunctionType.SIGMOID:
                self.activation = Sigmoid()
            elif function_type == FunctionType.TANH:
                self.activation = Tanh()
            else:
                self.activation = ReLU()
        else:
            self.activation = ReLU()

        self.neuron_num = neuron_num
        self.input_size = input_size
        he_std = math.sqrt(2 / input_size)
        self.parameters = [[random.gauss(0, he_std) for _ in range(input_size)] for _ in range(neuron_num)]
        self.bias = [[0] for _ in range(neuron_num)]
        self.z = None
        self.a = None
        self.first_moment_weight = [[0] * input_size for _ in range(neuron_num)]
        self.second_moment_weight = [[0] * input_size for _ in range(neuron_num)]
        self.first_moment_bias = [[0] for _ in range(neuron_num)]
        self.second_moment_bias = [[0] for _ in range(neuron_num)]
        self.dropout_mask = None
        self.W_node = None
        self.b_node = None
        self.z_node = None
        self.a_node = None

    def linearize(self, parameters, input, bias):
        parameter_input_product = custom_math.matrix_with_matrix_multiplication(parameters, input)
        dimension = len(parameter_input_product[0])
        broadcasted_matrix = [[0] * dimension for _ in range(len(parameter_input_product))]
        for row in range(len(broadcasted_matrix)):
            for col in range(len(broadcasted_matrix[0])):
                broadcasted_matrix[row][col] = bias[row][0]
        return custom_math.matrix_addition_and_sub(parameter_input_product, broadcasted_matrix, "add")

    def softmax(self, linear):
        linear_copy = [[0] * len(linear[0]) for _ in range(len(linear))]
        for col in range(len(linear[0])):
            exponential_sum = 0
            for row in range(len(linear)):
                exponential_sum += math.exp(linear[row][col])
            for row in range(len(linear)):
                linear_copy[row][col] = math.exp(linear[row][col]) / exponential_sum
        return linear_copy


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layers, dw, db):
        pass


class Adam(Optimizer):
    beta1 = 0.9
    beta2 = 0.999

    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.t = 0

    def update(self, layers, dw, db):
        self.t += 1
        bias_corr1 = 1 - self.beta1 ** self.t
        bias_corr2 = 1 - self.beta2 ** self.t

        for i in range(len(layers)):
            #Updating first moment for weights: m = beta1*m + (1-beta1)*dw
            term1_weight_moment_one = custom_math.scalar_multiply_matrix(layers[i].first_moment_weight, self.beta1)
            term2_weight_moment_one = custom_math.scalar_multiply_matrix(dw[i], 1 - self.beta1)
            layers[i].first_moment_weight = custom_math.matrix_addition_and_sub(term1_weight_moment_one, term2_weight_moment_one, "add")

            #Updating second moment for weights: v = beta2*v + (1-beta2)*dw^2
            term1_weight_moment_two = custom_math.scalar_multiply_matrix(layers[i].second_moment_weight, self.beta2)
            derivatives_weight_squared = custom_math.element_wise_multiplication(dw[i], dw[i])
            term2_weight_moment_two = custom_math.scalar_multiply_matrix(derivatives_weight_squared, 1 - self.beta2)
            layers[i].second_moment_weight = custom_math.matrix_addition_and_sub(term1_weight_moment_two, term2_weight_moment_two, "add")

            #Updating first moment for bias: m = beta1*m + (1-beta1)*db
            term1_bias_moment_one = custom_math.scalar_multiply_matrix(layers[i].first_moment_bias, self.beta1)
            term2_bias_moment_one = custom_math.scalar_multiply_matrix(db[i], 1 - self.beta1)
            layers[i].first_moment_bias = custom_math.matrix_addition_and_sub(term1_bias_moment_one, term2_bias_moment_one, "add")

            #Updating second moment for bias: v = beta2*v + (1-beta2)*db^2
            term1_bias_moment_two = custom_math.scalar_multiply_matrix(layers[i].second_moment_bias, self.beta2)
            derivatives_bias_squared = custom_math.element_wise_multiplication(db[i], db[i])
            term2_bias_moment_two = custom_math.scalar_multiply_matrix(derivatives_bias_squared, 1 - self.beta2)
            layers[i].second_moment_bias = custom_math.matrix_addition_and_sub(term1_bias_moment_two, term2_bias_moment_two, "add")

            #Bias correction: m_hat = m / (1 - beta^t)
            m_hat_weight = custom_math.scalar_multiply_matrix(layers[i].first_moment_weight, 1 / bias_corr1)
            v_hat_weight = custom_math.scalar_multiply_matrix(layers[i].second_moment_weight, 1 / bias_corr2)
            m_hat_bias = custom_math.scalar_multiply_matrix(layers[i].first_moment_bias, 1 / bias_corr1)
            v_hat_bias = custom_math.scalar_multiply_matrix(layers[i].second_moment_bias, 1 / bias_corr2)

            #Updating weight: params -= lr * m_hat / (sqrt(v_hat) + eps)
            root_v_weight = custom_math.element_wise_roots(v_hat_weight, 2)
            epsilon_matrix_w = [[1e-8] * len(v_hat_weight[0]) for _ in range(len(v_hat_weight))]
            denominator_weight = custom_math.matrix_addition_and_sub(root_v_weight, epsilon_matrix_w, "add")
            step_weight = custom_math.scalar_multiply_matrix(custom_math.element_wise_division_two_matrices(m_hat_weight, denominator_weight), self.learning_rate)
            layers[i].parameters = custom_math.matrix_addition_and_sub(layers[i].parameters, step_weight, "sub")

            #Updating bias
            root_v_bias = custom_math.element_wise_roots(v_hat_bias, 2)
            epsilon_matrix_b = [[1e-8] * len(v_hat_bias[0]) for _ in range(len(v_hat_bias))]
            denominator_bias = custom_math.matrix_addition_and_sub(root_v_bias, epsilon_matrix_b, "add")
            step_bias = custom_math.scalar_multiply_matrix(custom_math.element_wise_division_two_matrices(m_hat_bias, denominator_bias), self.learning_rate)
            layers[i].bias = custom_math.matrix_addition_and_sub(layers[i].bias, step_bias, "sub")


class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, layers, dw, db):
        for i in range(len(layers)):
            with_learning_rate_weight = custom_math.scalar_multiply_matrix(dw[i], self.learning_rate)
            with_learning_rate_bias = custom_math.scalar_multiply_matrix(db[i], self.learning_rate)
            layers[i].parameters = custom_math.matrix_addition_and_sub(layers[i].parameters, with_learning_rate_weight, "sub")
            layers[i].bias = custom_math.matrix_addition_and_sub(layers[i].bias, with_learning_rate_bias, "sub")


class Network:

    def __init__(self, layer_num, neurons_in_layers, initial_input, optimizer, dropout_rate, function_type=None, activation=None, weights_dir=None):
        self.number_of_layers = layer_num
        self.neurons_in_layers = neurons_in_layers
        self.initial_input = initial_input
        self.layers = []
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weights_dir = weights_dir

        for i in range(self.number_of_layers):
            if i == 0:
                layer = Layer(self.neurons_in_layers[i], len(self.initial_input), activation=activation, function_type=function_type)
            else:
                layer = Layer(self.neurons_in_layers[i], self.neurons_in_layers[i-1], activation=activation, function_type=function_type)
            self.layers.append(layer)

    #Expands the (neuron_num x 1) bias into (neuron_num x batch_size) so it can be added to
    #the pre-activation. The node is kept in the graph so its back() can reverse the broadcast
    #by summing the upstream gradient across the batch dimension, recovering a (neuron_num x 1) gradient.
    def _broadcast_bias(self, b_node, batch_size):
        data = [[b_node.matrix[r][0]] * batch_size for r in range(len(b_node.matrix))]
        res = Matrix(data, {b_node}, "broadcast")
        res.gradient = [[0] * batch_size for _ in range(len(b_node.matrix))]
        def back():
            #The gradient of a broadcast is the sum across the broadcasted dimension
            row_sums = [[sum(res.gradient[r])] for r in range(len(res.gradient))]
            b_node.gradient = custom_math.matrix_addition_and_sub(b_node.gradient, row_sums, "add")
        res.back = back
        return res

    #Forward pass builds the computation graph using Matrix nodes instead of raw lists.
    #Each layer's parameters and bias are wrapped as leaf Matrix nodes so their gradients are
    #automatically populated when topological_sort runs backward through the graph.
    def feedforward(self, layer_index, input_node):
        if layer_index >= len(self.layers):
            return

        layer = self.layers[layer_index]
        is_last = layer_index == len(self.layers) - 1
        batch_size = len(input_node.matrix[0])

        #Wrap parameters as leaf Matrix nodes and initialise their gradients as zero matrices
        #so matrix_addition_and_sub in each back() has a valid matrix to accumulate into
        W_node = Matrix(layer.parameters)
        b_node = Matrix(layer.bias)
        W_node.gradient = [[0] * len(layer.parameters[0]) for _ in range(len(layer.parameters))]
        b_node.gradient = [[0] for _ in range(len(layer.bias))]
        layer.W_node = W_node
        layer.b_node = b_node

        #input_node.gradient must be a matrix so matmul's back() can accumulate into it
        if not isinstance(input_node.gradient, list):
            input_node.gradient = [[0] * batch_size for _ in range(len(input_node.matrix))]

        #z = W @ X + b_broadcast — each operation creates a node that records its children
        b_broadcast = self._broadcast_bias(b_node, batch_size)
        matmul_node = W_node.__mul__(input_node)
        matmul_node.gradient = [[0] * batch_size for _ in range(layer.neuron_num)]
        z_node = matmul_node.__add__(b_broadcast)
        z_node.gradient = [[0] * batch_size for _ in range(layer.neuron_num)]
        layer.z_node = z_node
        layer.z = z_node.matrix

        if is_last:
            #Softmax stays outside the graph because column-wise normalisation requires a
            #broadcast division that Matrix does not yet support. Its gradient fuses cleanly
            #with cross-entropy in backward() so nothing is lost by keeping it separate.
            layer.a = layer.softmax(z_node.matrix)
            return self.feedforward(layer_index + 1, Matrix(layer.a))
        else:
            #apply_activation wraps any ActivationFunction (predefined or custom) into
            #a Matrix autograd node with the correct element-wise forward and backward pass.
            a_node = apply_activation(z_node, layer.activation)

            #Dropout is applied inside the graph so the mask correctly zeroes out the same
            #positions during the backward pass via element_wise_mult's back()
            if self.dropout_rate > 0:
                mask_data = [[1.0 if random.random() >= self.dropout_rate else 0.0
                              for _ in range(batch_size)] for _ in range(layer.neuron_num)]
                mask_node = Matrix(mask_data)
                a_node = a_node.element_wise_mult(mask_node)
                a_node = a_node.scalar_multiply(1.0 / (1.0 - self.dropout_rate))
                a_node.gradient = [[0] * batch_size for _ in range(layer.neuron_num)]

            layer.a_node = a_node
            layer.a = a_node.matrix
            return self.feedforward(layer_index + 1, a_node)

    def total_loss(self, output, loss_type, input_labels):
        total_loss = 0
        output_transpose = custom_math.transpose_matrix(output)
        input_labels_transpose = custom_math.transpose_matrix(input_labels)
        for row in range(len(output_transpose)):
            total_loss += loss_type(output_transpose[row], input_labels_transpose[row])
        total_loss /= len(input_labels_transpose)
        return total_loss

    #Replaces last_layer_backprop and previous_layer_backprop entirely.
    #The combined gradient of softmax + cross-entropy w.r.t the pre-softmax activations z is
    #(y_hat - y) / m. This is injected directly at the last layer's z_node so topological_sort
    #propagates it backward through every previous layer automatically via each node's back() closure.
    def backward(self, batch_labels):
        m = len(batch_labels[0])
        y_hat = self.layers[-1].a
        combined_gradient = [[(y_hat[r][c] - batch_labels[r][c]) / m
                               for c in range(len(y_hat[0]))]
                              for r in range(len(y_hat))]

        #Seed the gradient at the last layer's z_node and let the graph do the rest
        self.layers[-1].z_node.gradient = combined_gradient
        self.layers[-1].z_node.backprop()

    def train_loop(self, epochs, train_labels, batch_size):
        num_examples = len(train_labels[0])

        for epoch in range(epochs):
            indices = list(range(num_examples))
            random.shuffle(indices)
            num_batches = num_examples // batch_size

            for b in range(num_batches):
                batch_indices = indices[b * batch_size : (b + 1) * batch_size]
                batch_input  = [[row[j] for j in batch_indices] for row in self.initial_input]
                batch_labels = [[row[j] for j in batch_indices] for row in train_labels]

                #Forward pass — builds the computation graph for this batch
                self.feedforward(0, Matrix(batch_input))

                loss_fn = LossFunctions()
                current_loss = self.total_loss(self.layers[-1].a, loss_fn.cross_entropy_loss, batch_labels)
                print(f"Epoch {epoch}, Batch {b}, Loss: {current_loss}")

                #Backward pass — autograd propagates gradients through the graph
                self.backward(batch_labels)

                #Gradients now live in each layer's W_node and b_node after topological_sort
                weight_results = [self.layers[i].W_node.gradient for i in range(len(self.layers))]
                bias_results   = [self.layers[i].b_node.gradient for i in range(len(self.layers))]

                self.optimizer.update(self.layers, weight_results, bias_results)

    def save_weights(self):
        path = self.weights_dir if self.weights_dir is not None else "./weights.json"
        data = [{"parameters": layer.parameters, "bias": layer.bias} for layer in self.layers]
        with open(path, "w") as f:
            json.dump(data, f)

    def load_weights(self):
        path = self.weights_dir if self.weights_dir is not None else "./weights.json"
        with open(path, "r") as f:
            data = json.load(f)
        for i, layer_data in enumerate(data):
            self.layers[i].parameters = layer_data["parameters"]
            self.layers[i].bias = layer_data["bias"]

    #Runs a forward pass on x_test and returns the percentage of correctly classified examples.
    #The predicted class is the argmax of each output column, same for the actual label.
    def accuracy(self, x_test, y_test):
        saved_dropout = self.dropout_rate
        self.dropout_rate = 0
        self.feedforward(0, Matrix(x_test))
        self.dropout_rate = saved_dropout
        output = self.layers[-1].a
        correct = 0
        for col in range(len(output[0])):
            pred   = max(range(len(output)),  key=lambda row: output[row][col])
            actual = max(range(len(y_test)),  key=lambda row: y_test[row][col])
            if pred == actual:
                correct += 1
        return (correct / len(output[0])) * 100
