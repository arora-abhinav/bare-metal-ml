#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <functional>
#include <limits>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include "../custom_math.cpp"

using namespace std;

typedef vector<double> Vec;
typedef vector<vector<double>> Mat;

//Reads a big-endian unsigned 32-bit integer from a binary file, matching Python's struct.unpack(">I")
uint32_t read_be_uint32(ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | bytes[3];
}

//This is to one-hot encode input labels whenever softmax regression is used.
//The class number simply is the number of classes
Mat vectorize_labels(vector<uint8_t> input_labels, int class_number) {
    //The one-hot encoded arrays are being stored in a matrix where each column corresponds to the number of classes
    //And the row corresponds to which class-type. This is consistent with the dimensions of the output of the final layer
    Mat res(class_number, Vec(input_labels.size(), 0.0));
    for (int i = 0; i < (int)input_labels.size(); i++) {
        res[input_labels[i]][i] = 1.0;
    }
    return res;
}

class ActivationFunctions {
public:
    double sigmoid(double input_val) {
        double exponent = exp(-input_val);
        return 1.0 / (1.0 + exponent);
    }

    double ReLU(double input_val) {
        if (input_val > 0) return input_val;
        return 0.0;
    }
};

enum class FunctionType { RELU, SIGMOID };

class Layer {
public:
    FunctionType function_type;
    int neuron_num;
    int input_size;
    Mat parameters;
    Mat bias;
    //Linear part and result of the activation being stored directly as a property of the layer itself
    Mat z;
    Mat a;

    Layer(int neuron_num, int input_size, FunctionType function_type) {
        this->function_type = function_type;
        this->neuron_num = neuron_num;
        this->input_size = input_size;

        //Generates a random number in the normal distribution with mean 0 and std 0.01. This results in values between
        //-0.03 and 0.03. This is required since we want to break symmetry in each layer of a neural network and not
        //make them identical
        static mt19937 gen(random_device{}());
        static normal_distribution<double> gauss(0.0, 0.01);

        parameters.resize(neuron_num, Vec(input_size));
        for (int i = 0; i < neuron_num; i++) {
            for (int j = 0; j < input_size; j++) {
                parameters[i][j] = gauss(gen);
            }
        }

        bias.resize(neuron_num, Vec(1, 0.0));
    }

    double execute_function(double input_val) {
        ActivationFunctions functions;
        if (function_type == FunctionType::RELU) {
            return functions.ReLU(input_val);
        }
        if (function_type == FunctionType::SIGMOID) {
            return functions.sigmoid(input_val);
        }
        return 0.0;
    }

    //Returns the omega * x + b
    //Bias automatically gets broadcasted in this function
    Mat linearize(Mat parameters, Mat input, Mat bias) {
        Mat parameter_input_product = matrix_with_matrix_multiplication(parameters, input);
        //Broadcast
        //This is the number of columns of the result, which equals the number of training examples
        int dimension = parameter_input_product[0].size();
        Mat broadcasted_matrix(parameter_input_product.size(), Vec(dimension, 0.0));
        for (int row = 0; row < (int)broadcasted_matrix.size(); row++) {
            for (int col = 0; col < (int)broadcasted_matrix[0].size(); col++) {
                broadcasted_matrix[row][col] = bias[row][0];
            }
        }
        return matrix_addition_and_sub(parameter_input_product, broadcasted_matrix, "add");
    }

    //The softmax function is only for the last layer's output, by default it will be set to false. However, the
    //underlying hypothesis function will not change
    Mat hypothesis(Mat linear, bool softmax = false) {
        Mat linear_copy(linear.size(), Vec(linear[0].size(), 0.0));
        for (int row = 0; row < (int)linear.size(); row++) {
            for (int col = 0; col < (int)linear[0].size(); col++) {
                linear_copy[row][col] = execute_function(linear[row][col]);
            }
        }

        if (!softmax) {
            return linear_copy;
        } else {
            for (int col = 0; col < (int)linear[0].size(); col++) {
                //Calculating the total exponential sum for each output
                double exponential_sum = 0.0;
                for (int row = 0; row < (int)linear.size(); row++) {
                    exponential_sum += exp(linear_copy[row][col]);
                }
                //Applying the softmax formula
                for (int row = 0; row < (int)linear.size(); row++) {
                    linear_copy[row][col] = exp(linear_copy[row][col]) / exponential_sum;
                }
            }
        }

        return linear_copy;
    }
};

class LossFunctions {
public:
    //This is a function that will add an epsilon value (extremely small) to prevent from obtaining the log(0) in any calculation
    Vec regularise(Vec vector, double epsilon = 1e-5) {
        Vec vector_copy(vector.size(), 0.0);
        for (int i = 0; i < (int)vector.size(); i++) {
            vector_copy[i] = vector[i] + epsilon;
        }
        return vector_copy;
    }

    //This is the cross entropy function required.
    //y_hat is the final prediction vector and y is the actual label vector
    double cross_entropy_loss(Vec y_hat, Vec y) {
        Vec regularised_predictions = regularise(y_hat);
        for (int i = 0; i < (int)regularised_predictions.size(); i++) {
            regularised_predictions[i] = log(regularised_predictions[i]) * y[i];
        }
        double res = 0.0;
        for (double val : regularised_predictions) res += val;
        return -res;
    }
};

class Network {
public:
    int number_of_layers;
    vector<int> neurons_in_layers;
    Mat initial_input;
    vector<Layer> layers;

    Network(int layer_num, vector<int> neurons_in_layers, Mat initial_input, FunctionType function_type) {
        this->number_of_layers = layer_num;
        this->neurons_in_layers = neurons_in_layers;
        this->initial_input = initial_input;

        //Initialising the layers
        for (int i = 0; i < number_of_layers; i++) {
            //This is specifically for the first layer. This is because the input_size is the dimension of the vector of each training example
            if (i == 0) {
                layers.push_back(Layer(neurons_in_layers[i], initial_input.size(), function_type));
            }
            //For the other layers, the input size is the number of neurons of the previous layer since each neuron outputs a single number
            else {
                layers.push_back(Layer(neurons_in_layers[i], neurons_in_layers[i - 1], function_type));
            }
        }
    }

    //This is the feedforward function. This will be a recursive function.
    //The layer_index specifies which layer in layers and input specifies the input for each layer
    void feedforward(int layer_index, Mat input) {
        //Base case
        if (layer_index >= (int)layers.size()) return;
        Layer& layer = layers[layer_index];
        Mat linear_res = layer.linearize(layer.parameters, input, layer.bias);
        Mat output;
        if (layer_index == (int)layers.size() - 1) {
            output = layer.hypothesis(linear_res, true);
        } else {
            output = layer.hypothesis(linear_res);
        }
        //Useful for caching results
        layer.a = output;
        layer.z = linear_res;
        //The input of the next layer becomes the output of the current layer
        feedforward(layer_index + 1, output);
    }

    //The total loss function is the average of the loss functions across all training examples.
    //The output is the output of the final layer
    double total_loss(Mat output, function<double(Vec, Vec)> loss_type, Mat input_labels) {
        double total = 0.0;
        //The output has dimension (number of neurons in last layer, training examples) but this makes it hard to iterate over each column
        //Therefore, the transpose allows us to iterate row by row
        //Same logic for input_labels since they were one-hot encoded to be in the same dimension as the output
        Mat output_transpose = transpose_matrix(output);
        Mat input_labels_transpose = transpose_matrix(input_labels);
        for (int row = 0; row < (int)output_transpose.size(); row++) {
            total += loss_type(output_transpose[row], input_labels_transpose[row]);
        }
        total /= input_labels_transpose.size();
        return total;
    }

    //The following backprop functions are hardcoded for cross entropy. It is not practical to have such hardcoded backprop functions
    //and so an autograd engine will be implemented later on. There is a possibility that the previous_layer is null and so the input is just the initial input
    pair<Mat, Mat> last_layer_backprop(Mat labels, Layer& final_layer, Layer* prev_layer = nullptr) {
        Mat prev_activation_transpose;
        if (prev_layer != nullptr) {
            prev_activation_transpose = transpose_matrix(prev_layer->a);
        } else {
            prev_activation_transpose = transpose_matrix(initial_input);
        }

        Mat term_one = matrix_addition_and_sub(labels, final_layer.a, "sub");
        Mat final_prod = matrix_with_matrix_multiplication(term_one, prev_activation_transpose);
        //The total loss is the average of losses across each training example
        Mat res = scalar_multiply_matrix(final_prod, -1.0 / labels[0].size());
        //This is the product that will be backpropagated. According to calculations, the product backpropagated to the previous
        //layer (W[l-1]) is the same as dJ/dW[l] without A[l-1]T.
        Mat product_two = scalar_multiply_matrix(term_one, -1.0 / labels[0].size());
        return {res, product_two};
    }

    //This is the general pattern for the backprop for previous layers.
    pair<Mat, Mat> previous_layer_backprop(Layer& current_layer, Layer& next_layer, Mat previous_product, Layer* previous_layer = nullptr) {
        Mat next_layer_parameter_transpose = transpose_matrix(next_layer.parameters);
        Mat multiplied_term_one = matrix_with_matrix_multiplication(next_layer_parameter_transpose, previous_product);
        Mat product_one = element_wise_multiplication(multiplied_term_one, current_layer.a);
        Mat matrix_of_ones(current_layer.a.size(), Vec(current_layer.a[0].size(), 1.0));
        Mat one_minus_a = matrix_addition_and_sub(matrix_of_ones, current_layer.a, "sub");
        Mat product_two = element_wise_multiplication(product_one, one_minus_a);
        Mat previous_layer_activation_transpose;
        if (previous_layer != nullptr) {
            previous_layer_activation_transpose = transpose_matrix(previous_layer->a);
        } else {
            previous_layer_activation_transpose = transpose_matrix(initial_input);
        }
        //res is dJ/dW[l] of the current layer whereas product_two is the actual product that will be backpropagated.
        //product_two is also dJ/dB[l]
        Mat res = matrix_with_matrix_multiplication(product_two, previous_layer_activation_transpose);
        return {res, product_two};
    }

    //The training loop is as follows:
    //1) Feedforward and store produced activation, parameters and bias in the layer
    //2) Backprop and update each parameter
    //3) Repeat until parameter convergence
    void train_loop(int epochs, double learning_rate, Mat train_labels) {
        LossFunctions loss_fn;
        for (int epoch = 0; epoch < epochs; epoch++) {
            feedforward(0, initial_input);
            double current_loss = total_loss(layers.back().a, [&](Vec y_hat, Vec y) {
                return loss_fn.cross_entropy_loss(y_hat, y);
            }, train_labels);
            cout << "Epoch " << epoch << ", Loss: " << current_loss << endl;

            //Storing the current epoch's calculated weight and bias results
            vector<Mat> weight_results(layers.size());
            vector<Mat> bias_results(layers.size());

            //The backprop_prod is the same as dJ/dB[l]
            auto [last_layer_res, backprop_prod] = (layers.size() > 1) ?
                last_layer_backprop(train_labels, layers.back(), &layers[layers.size() - 2]) :
                last_layer_backprop(train_labels, layers.back());

            weight_results.back() = last_layer_res;
            //Summing backprop_prod across columns to get (K, 1) bias gradient
            Mat last_bias_grad(backprop_prod.size(), Vec(1, 0.0));
            for (int row = 0; row < (int)backprop_prod.size(); row++) {
                double s = 0.0;
                for (double val : backprop_prod[row]) s += val;
                last_bias_grad[row][0] = s;
            }
            bias_results.back() = last_bias_grad;

            for (int i = (int)layers.size() - 2; i >= 0; i--) {
                auto [res, new_backprop_prod] = (i > 0) ?
                    previous_layer_backprop(layers[i], layers[i + 1], backprop_prod, &layers[i - 1]) :
                    previous_layer_backprop(layers[i], layers[i + 1], backprop_prod);
                backprop_prod = new_backprop_prod;

                Mat bias_grad(backprop_prod.size(), Vec(1, 0.0));
                for (int row = 0; row < (int)backprop_prod.size(); row++) {
                    double s = 0.0;
                    for (double val : backprop_prod[row]) s += val;
                    bias_grad[row][0] = s;
                }
                bias_results[i] = bias_grad;
                weight_results[i] = res;
            }

            //Updating the parameters according to batch gradient descent
            for (int i = 0; i < (int)layers.size(); i++) {
                Mat with_learning_rate_weight = scalar_multiply_matrix(weight_results[i], learning_rate);
                Mat with_learning_rate_bias = scalar_multiply_matrix(bias_results[i], learning_rate);
                layers[i].parameters = matrix_addition_and_sub(layers[i].parameters, with_learning_rate_weight, "sub");
                layers[i].bias = matrix_addition_and_sub(layers[i].bias, with_learning_rate_bias, "sub");
            }
        }
    }
};

int main(int argc, char** argv) {
    //Parsing the binary format
    ifstream img_file("MNIST handwritten /train-images-idx3-ubyte/train-images-idx3-ubyte", ios::binary);
    uint32_t magic = read_be_uint32(img_file);
    uint32_t num   = read_be_uint32(img_file);
    uint32_t rows  = read_be_uint32(img_file);
    uint32_t cols  = read_be_uint32(img_file);

    vector<uint8_t> raw_images(num * rows * cols);
    img_file.read(reinterpret_cast<char*>(raw_images.data()), raw_images.size());
    img_file.close();

    //Loading in the labels
    ifstream lbl_file("MNIST handwritten /train-labels-idx1-ubyte/train-labels-idx1-ubyte", ios::binary);
    uint32_t lbl_magic = read_be_uint32(lbl_file);
    uint32_t lbl_num   = read_be_uint32(lbl_file);

    vector<uint8_t> labels(lbl_num);
    lbl_file.read(reinterpret_cast<char*>(labels.data()), labels.size());
    lbl_file.close();

    Mat vectorized_labels = vectorize_labels(labels, 10);

    int img_dimension = rows * cols;
    //Each image flattened into a vector and put into a matrix
    Mat image_input_matrix(img_dimension, Vec(num, 0.0));

    for (int column = 0; column < (int)num; column++) {
        for (int row = 0; row < (int)(rows * cols); row++) {
            //Normalizing to fit every value between 0 and 1 to solve vanishing gradients
            image_input_matrix[row][column] = raw_images[column * (rows * cols) + row] / 255.0;
        }
    }

    int num_examples = image_input_matrix[0].size();
    int train_size = (int)(0.8 * num_examples);

    //Shuffle indices
    vector<int> indices(num_examples);
    iota(indices.begin(), indices.end(), 0);
    mt19937 gen(random_device{}());
    shuffle(indices.begin(), indices.end(), gen);

    vector<int> train_indices(indices.begin(), indices.begin() + train_size);
    vector<int> test_indices(indices.begin() + train_size, indices.end());

    //Split inputs — selecting columns corresponding to each split
    Mat x_train(image_input_matrix.size(), Vec(train_size, 0.0));
    Mat x_test(image_input_matrix.size(), Vec(test_indices.size(), 0.0));
    for (int row = 0; row < (int)image_input_matrix.size(); row++) {
        for (int i = 0; i < train_size; i++) {
            x_train[row][i] = image_input_matrix[row][train_indices[i]];
        }
        for (int i = 0; i < (int)test_indices.size(); i++) {
            x_test[row][i] = image_input_matrix[row][test_indices[i]];
        }
    }

    //Split labels the same way
    Mat y_train(vectorized_labels.size(), Vec(train_size, 0.0));
    Mat y_test(vectorized_labels.size(), Vec(test_indices.size(), 0.0));
    for (int row = 0; row < (int)vectorized_labels.size(); row++) {
        for (int i = 0; i < train_size; i++) {
            y_train[row][i] = vectorized_labels[row][train_indices[i]];
        }
        for (int i = 0; i < (int)test_indices.size(); i++) {
            y_test[row][i] = vectorized_labels[row][test_indices[i]];
        }
    }

    Network MNIST_Network(4, {784, 128, 64, 10}, x_train, FunctionType::SIGMOID);
    MNIST_Network.train_loop(1000, 1, y_train);

    return 0;
}
