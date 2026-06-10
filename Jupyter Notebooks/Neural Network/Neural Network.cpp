#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <functional>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <sstream>
#include "../../custom_math.cpp"
#include "autograd.hpp"

using namespace std;
typedef vector<double> Vec;
typedef vector<vector<double>> Mat;

uint32_t read_be_uint32(ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | bytes[3];
}

Mat vectorize_labels(vector<uint8_t> input_labels, int class_number) {
    Mat res(class_number, Vec(input_labels.size(), 0.0));
    for (int i = 0; i < (int)input_labels.size(); i++)
        res[input_labels[i]][i] = 1.0;
    return res;
}

//Users inherit from ActivationFunction and implement forward() and derivative() at scalar level.
//The framework applies both element-wise across the matrix automatically via apply_activation().
//Each subclass represents one activation function along with its derivative — keeping them
//together ensures the forward and backward passes always stay in sync.
class ActivationFunction {
public:
    virtual double forward(double x)    = 0;
    virtual double derivative(double x) = 0;
    virtual ~ActivationFunction()       = default;
};

//Predefined activations — users can use these directly or as reference implementations
class ReLU : public ActivationFunction {
public:
    double forward(double x)    override { return x > 0.0 ? x : 0.0; }
    double derivative(double x) override { return x > 0.0 ? 1.0 : 0.0; }
};

class Sigmoid : public ActivationFunction {
public:
    double forward(double x) override { return 1.0 / (1.0 + exp(-x)); }
    double derivative(double x) override {
        double s = forward(x);
        return s * (1.0 - s);
    }
};

class Tanh : public ActivationFunction {
public:
    double forward(double x)    override { return tanh(x); }
    double derivative(double x) override { return 1.0 - tanh(x) * tanh(x); }
};

//Convenience enum for selecting predefined activations without constructing them manually.
//Pass a custom ActivationFunction* to the Network constructor to override this entirely.
enum class FunctionType { RELU, SIGMOID, TANH };

//Wraps any ActivationFunction into a Matrix autograd node.
//forward() is applied element-wise to build the output matrix.
//back() uses derivative() element-wise and multiplies by the upstream gradient.
MatrixPtr apply_activation(MatrixPtr z_node, shared_ptr<ActivationFunction> act) {
    int rows = z_node->matrix.size(), cols = z_node->matrix[0].size();
    Mat result(rows, Vec(cols, 0.0));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            result[r][c] = act->forward(z_node->matrix[r][c]);
    auto res = make_shared<Matrix>(result, set<ElemPtr>{z_node}, "activation");
    weak_ptr<Matrix> res_weak = res;
    res->back = [res_weak, z_node, act]() {
        auto res = res_weak.lock(); if (!res) return;
        //Apply the scalar derivative element-wise, then multiply by the upstream gradient
        int rows = z_node->matrix.size(), cols = z_node->matrix[0].size();
        Mat deriv(rows, Vec(cols, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                deriv[r][c] = act->derivative(z_node->matrix[r][c]);
        z_node->gradient = matrix_addition_and_sub(z_node->gradient,
            element_wise_multiplication(deriv, res->gradient), "add");
    };
    return res;
}

class Layer {
public:
    shared_ptr<ActivationFunction> activation;
    int neuron_num;
    int input_size;
    Mat parameters;
    Mat bias;
    Mat z;
    Mat a;
    Mat first_moment_weight;
    Mat second_moment_weight;
    Mat first_moment_bias;
    Mat second_moment_bias;
    Mat dropout_mask;
    MatrixPtr W_node;
    MatrixPtr b_node;
    MatrixPtr z_node;
    MatrixPtr a_node;

    Layer(int neuron_num, int input_size, shared_ptr<ActivationFunction> activation) {
        this->activation = activation;
        this->neuron_num = neuron_num;
        this->input_size = input_size;

        static mt19937 gen(random_device{}());
        double he_std = sqrt(2.0 / input_size);
        normal_distribution<double> gauss(0.0, he_std);

        parameters.resize(neuron_num, Vec(input_size));
        for (int i = 0; i < neuron_num; i++)
            for (int j = 0; j < input_size; j++)
                parameters[i][j] = gauss(gen);

        bias.resize(neuron_num, Vec(1, 0.0));

        first_moment_weight.assign(neuron_num, Vec(input_size, 0.0));
        second_moment_weight.assign(neuron_num, Vec(input_size, 0.0));
        first_moment_bias.assign(neuron_num, Vec(1, 0.0));
        second_moment_bias.assign(neuron_num, Vec(1, 0.0));
    }

    Mat linearize(Mat params, Mat input, Mat b) {
        Mat product = matrix_with_matrix_multiplication(params, input);
        int cols = product[0].size();
        Mat broadcasted(product.size(), Vec(cols, 0.0));
        for (int row = 0; row < (int)broadcasted.size(); row++)
            for (int col = 0; col < cols; col++)
                broadcasted[row][col] = b[row][0];
        return matrix_addition_and_sub(product, broadcasted, "add");
    }

    //Only used for the output layer — applies softmax directly to the logits.
    //Hidden layer activations go through apply_activation() in the autograd graph instead.
    Mat softmax(Mat linear) {
        Mat result(linear.size(), Vec(linear[0].size(), 0.0));
        for (int col = 0; col < (int)linear[0].size(); col++) {
            double exp_sum = 0.0;
            for (int row = 0; row < (int)linear.size(); row++)
                exp_sum += exp(linear[row][col]);
            for (int row = 0; row < (int)linear.size(); row++)
                result[row][col] = exp(linear[row][col]) / exp_sum;
        }
        return result;
    }
};

class LossFunctions {
public:
    Vec regularise(Vec vector, double epsilon = 1e-7) {
        Vec copy(vector.size());
        for (int i = 0; i < (int)vector.size(); i++)
            copy[i] = max(epsilon, min(1.0 - epsilon, vector[i]));
        return copy;
    }

    double cross_entropy_loss(Vec y_hat, Vec y) {
        Vec reg = regularise(y_hat);
        for (int i = 0; i < (int)reg.size(); i++)
            reg[i] = log(reg[i]) * y[i];
        double res = 0.0;
        for (double val : reg) res += val;
        return -res;
    }
};

class Optimizer {
public:
    double learning_rate;
    Optimizer(double lr) : learning_rate(lr) {}
    virtual void update(vector<Layer>& layers, vector<Mat>& dw, vector<Mat>& db) = 0;
    virtual ~Optimizer() = default;
};

class Adam : public Optimizer {
public:
    int t = 0;
    static constexpr double beta1 = 0.9;
    static constexpr double beta2 = 0.999;

    Adam(double lr) : Optimizer(lr) {}

    void update(vector<Layer>& layers, vector<Mat>& dw, vector<Mat>& db) override {
        t++;
        double bias_corr1 = 1.0 - pow(beta1, t);
        double bias_corr2 = 1.0 - pow(beta2, t);

        for (int i = 0; i < (int)layers.size(); i++) {
            // First moment: m = beta1*m + (1-beta1)*dw
            Mat t1_w1 = scalar_multiply_matrix(layers[i].first_moment_weight, beta1);
            Mat t2_w1 = scalar_multiply_matrix(dw[i], 1.0 - beta1);
            layers[i].first_moment_weight = matrix_addition_and_sub(t1_w1, t2_w1, "add");

            // Second moment: v = beta2*v + (1-beta2)*dw^2
            Mat t1_w2 = scalar_multiply_matrix(layers[i].second_moment_weight, beta2);
            Mat dw_sq = element_wise_multiplication(dw[i], dw[i]);
            Mat t2_w2 = scalar_multiply_matrix(dw_sq, 1.0 - beta2);
            layers[i].second_moment_weight = matrix_addition_and_sub(t1_w2, t2_w2, "add");

            // First moment bias: m = beta1*m + (1-beta1)*db
            Mat t1_b1 = scalar_multiply_matrix(layers[i].first_moment_bias, beta1);
            Mat t2_b1 = scalar_multiply_matrix(db[i], 1.0 - beta1);
            layers[i].first_moment_bias = matrix_addition_and_sub(t1_b1, t2_b1, "add");

            // Second moment bias: v = beta2*v + (1-beta2)*db^2
            Mat t1_b2 = scalar_multiply_matrix(layers[i].second_moment_bias, beta2);
            Mat db_sq = element_wise_multiplication(db[i], db[i]);
            Mat t2_b2 = scalar_multiply_matrix(db_sq, 1.0 - beta2);
            layers[i].second_moment_bias = matrix_addition_and_sub(t1_b2, t2_b2, "add");

            // Bias correction: m_hat = m / (1 - beta^t)
            Mat m_hat_w = scalar_multiply_matrix(layers[i].first_moment_weight, 1.0 / bias_corr1);
            Mat v_hat_w = scalar_multiply_matrix(layers[i].second_moment_weight, 1.0 / bias_corr2);
            Mat m_hat_b = scalar_multiply_matrix(layers[i].first_moment_bias, 1.0 / bias_corr1);
            Mat v_hat_b = scalar_multiply_matrix(layers[i].second_moment_bias, 1.0 / bias_corr2);

            // Update weights: params -= lr * m_hat / (sqrt(v_hat) + eps)
            Mat root_v_w = element_wise_roots(v_hat_w, 2.0);
            Mat eps_w(root_v_w.size(), Vec(root_v_w[0].size(), 1e-8));
            Mat denom_w = matrix_addition_and_sub(root_v_w, eps_w, "add");
            Mat step_w = scalar_multiply_matrix(element_wise_division_two_matrices(m_hat_w, denom_w), learning_rate);
            layers[i].parameters = matrix_addition_and_sub(layers[i].parameters, step_w, "sub");

            // Update bias
            Mat root_v_b = element_wise_roots(v_hat_b, 2.0);
            Mat eps_b(root_v_b.size(), Vec(root_v_b[0].size(), 1e-8));
            Mat denom_b = matrix_addition_and_sub(root_v_b, eps_b, "add");
            Mat step_b = scalar_multiply_matrix(element_wise_division_two_matrices(m_hat_b, denom_b), learning_rate);
            layers[i].bias = matrix_addition_and_sub(layers[i].bias, step_b, "sub");
        }
    }
};

class SGD : public Optimizer {
public:
    SGD(double lr) : Optimizer(lr) {}

    void update(vector<Layer>& layers, vector<Mat>& dw, vector<Mat>& db) override {
        for (int i = 0; i < (int)layers.size(); i++) {
            Mat step_w = scalar_multiply_matrix(dw[i], learning_rate);
            Mat step_b = scalar_multiply_matrix(db[i], learning_rate);
            layers[i].parameters = matrix_addition_and_sub(layers[i].parameters, step_w, "sub");
            layers[i].bias = matrix_addition_and_sub(layers[i].bias, step_b, "sub");
        }
    }
};

class Network {
public:
    int number_of_layers;
    vector<int> neurons_in_layers;
    Mat initial_input;
    vector<Layer> layers;
    Optimizer* optimizer;
    Mat current_batch;
    double dropout_rate;

    //function_type selects a predefined activation. Pass a custom ActivationFunction* to
    //override it entirely — the custom pointer takes priority if both are provided.
    Network(int layer_num, vector<int> neurons_in_layers, Mat initial_input,
            FunctionType function_type, Optimizer* optimizer,
            double dropout_rate = 0.0, ActivationFunction* custom_activation = nullptr) {
        this->number_of_layers = layer_num;
        this->neurons_in_layers = neurons_in_layers;
        this->initial_input = initial_input;
        this->optimizer = optimizer;
        this->dropout_rate = dropout_rate;

        //Resolve activation: custom takes priority, otherwise use the predefined enum
        shared_ptr<ActivationFunction> act;
        if (custom_activation) {
            //User manages lifetime of custom_activation — wrap with no-op deleter
            act = shared_ptr<ActivationFunction>(custom_activation, [](ActivationFunction*){});
        } else {
            switch (function_type) {
                case FunctionType::SIGMOID: act = make_shared<Sigmoid>(); break;
                case FunctionType::TANH:    act = make_shared<Tanh>();    break;
                default:                   act = make_shared<ReLU>();     break;
            }
        }

        for (int i = 0; i < number_of_layers; i++) {
            if (i == 0)
                layers.push_back(Layer(neurons_in_layers[i], initial_input.size(), act));
            else
                layers.push_back(Layer(neurons_in_layers[i], neurons_in_layers[i - 1], act));
        }

        //Create W_node and b_node once — they persist across batches.
        //Each batch only resets their gradients and syncs the matrix data from layer.parameters/bias
        //after the optimizer updates them, avoiding a full heap allocation + copy per batch.
        for (auto& layer : layers) {
            layer.W_node = make_shared<Matrix>(layer.parameters);
            layer.b_node = make_shared<Matrix>(layer.bias);
        }
    }

    //Expands the (neuron_num x 1) bias into (neuron_num x batch_size) so it can be added to
    //the pre-activation. The node is kept in the graph so its back() can reverse the broadcast
    //by summing the upstream gradient across the batch dimension, recovering a (neuron_num x 1) gradient.
    MatrixPtr broadcast_bias(MatrixPtr b_node, int batch_size) {
        int rows = b_node->matrix.size();
        Mat data(rows, Vec(batch_size, 0.0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < batch_size; c++)
                data[r][c] = b_node->matrix[r][0];
        auto res = make_shared<Matrix>(data, set<ElemPtr>{b_node}, "broadcast");
        weak_ptr<Matrix> res_weak = res;
        res->back = [res_weak, b_node, batch_size]() {
            auto res = res_weak.lock(); if (!res) return;
            //The gradient of a broadcast is the sum across the broadcasted dimension
            int rows = res->gradient.size();
            Mat row_sums(rows, Vec(1, 0.0));
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < batch_size; c++)
                    row_sums[r][0] += res->gradient[r][c];
            b_node->gradient = matrix_addition_and_sub(b_node->gradient, row_sums, "add");
        };
        return res;
    }

    //Forward pass builds the computation graph using Matrix nodes instead of raw lists.
    //Each layer's parameters and bias are wrapped as leaf Matrix nodes so their gradients are
    //automatically populated when backprop() runs backward through the graph.
    void feedforward(int layer_index, MatrixPtr input_node) {
        if (layer_index >= (int)layers.size()) return;

        Layer& layer = layers[layer_index];
        bool is_last = (layer_index == (int)layers.size() - 1);
        int batch_size = input_node->matrix[0].size();

        //Sync latest parameter values, reset gradients, and clear stale graph edges
        //from the previous batch so old nodes are freed and don't accumulate in memory.
        layer.W_node->matrix   = layer.parameters;
        layer.W_node->gradient = Mat(layer.neuron_num, Vec(layer.input_size, 0.0));
        layer.W_node->children = {};
        layer.b_node->matrix   = layer.bias;
        layer.b_node->gradient = Mat(layer.neuron_num, Vec(1, 0.0));
        layer.b_node->children = {};

        //z = W @ X + b_broadcast — each operation creates a node that records its children
        MatrixPtr b_broadcast = broadcast_bias(layer.b_node, batch_size);
        MatrixPtr matmul_node = static_pointer_cast<Matrix>(layer.W_node->mul(input_node));
        layer.z_node          = static_pointer_cast<Matrix>(matmul_node->add(b_broadcast));
        layer.z               = layer.z_node->matrix;

        if (is_last) {
            //Softmax stays outside the graph because column-wise normalisation requires a
            //broadcast division that Matrix does not yet support. Its gradient fuses cleanly
            //with cross-entropy in backward() so nothing is lost by keeping it separate.
            layer.a = layer.softmax(layer.z_node->matrix);
            return feedforward(layer_index + 1, make_shared<Matrix>(layer.a));
        } else {
            //apply_activation wraps any ActivationFunction — predefined or custom — into
            //a Matrix autograd node with the correct element-wise forward and backward pass.
            MatrixPtr a_node = apply_activation(layer.z_node, layer.activation);

            //Dropout is applied inside the graph so the mask correctly zeroes out the same
            //positions during the backward pass via element_wise_mult's back()
            if (dropout_rate > 0.0) {
                static mt19937 rng(random_device{}());
                uniform_real_distribution<double> dist(0.0, 1.0);
                Mat mask_data(layer.neuron_num, Vec(batch_size, 0.0));
                for (int r = 0; r < layer.neuron_num; r++)
                    for (int c = 0; c < batch_size; c++)
                        mask_data[r][c] = dist(rng) >= dropout_rate ? 1.0 : 0.0;
                auto mask_node = make_shared<Matrix>(mask_data);
                a_node = a_node->element_wise_mult(mask_node);
                a_node = a_node->scalar_multiply(1.0 / (1.0 - dropout_rate));
                a_node->gradient = Mat(layer.neuron_num, Vec(batch_size, 0.0));
            }

            layer.a_node = a_node;
            layer.a = a_node->matrix;
            return feedforward(layer_index + 1, a_node);
        }
    }

    double total_loss(Mat output, function<double(Vec, Vec)> loss_type, Mat input_labels) {
        double total = 0.0;
        Mat output_T = transpose_matrix(output);
        Mat labels_T = transpose_matrix(input_labels);
        for (int row = 0; row < (int)output_T.size(); row++)
            total += loss_type(output_T[row], labels_T[row]);
        total /= labels_T.size();
        return total;
    }

    //Replaces last_layer_backprop and previous_layer_backprop entirely.
    //The combined gradient of softmax + cross-entropy w.r.t the pre-softmax activations z is
    //(y_hat - y) / m. This is injected directly at the last layer's z_node so backprop()
    //propagates it backward through every previous layer automatically via each node's back() closure.
    void backward(Mat batch_labels) {
        int m = batch_labels[0].size();
        Mat& y_hat = layers.back().a;
        Mat combined_gradient(y_hat.size(), Vec(y_hat[0].size(), 0.0));
        for (int r = 0; r < (int)y_hat.size(); r++)
            for (int c = 0; c < (int)y_hat[0].size(); c++)
                combined_gradient[r][c] = (y_hat[r][c] - batch_labels[r][c]) / m;

        //Seed the gradient at the last layer's z_node and let the graph do the rest
        layers.back().z_node->gradient = combined_gradient;
        layers.back().z_node->backprop();
    }

    void train_loop(int epochs, Mat train_labels, int batch_size) {
        LossFunctions loss_fn;
        int num_examples = initial_input[0].size();
        static mt19937 rng(random_device{}());

        for (int epoch = 0; epoch < epochs; epoch++) {
            vector<int> indices(num_examples);
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), rng);

            int num_batches = num_examples / batch_size;
            for (int b = 0; b < num_batches; b++) {
                int start = b * batch_size;

                Mat batch_input(initial_input.size(), Vec(batch_size, 0.0));
                Mat batch_labels(train_labels.size(), Vec(batch_size, 0.0));
                for (int row = 0; row < (int)initial_input.size(); row++)
                    for (int j = 0; j < batch_size; j++)
                        batch_input[row][j] = initial_input[row][indices[start + j]];
                for (int row = 0; row < (int)train_labels.size(); row++)
                    for (int j = 0; j < batch_size; j++)
                        batch_labels[row][j] = train_labels[row][indices[start + j]];

                //Forward pass — builds the computation graph for this batch.
                //Constructor auto-initialises input_node gradient to zeros so mul's back()
                //can safely accumulate into it even though we don't need dL/dX for the input.
                feedforward(0, make_shared<Matrix>(batch_input));

                double current_loss = total_loss(layers.back().a, [&](Vec y_hat, Vec y) {
                    return loss_fn.cross_entropy_loss(y_hat, y);
                }, batch_labels);
                cout << "Epoch " << epoch << ", Batch " << b << ", Loss: " << current_loss << endl;

                //Backward pass — autograd propagates gradients through the graph
                backward(batch_labels);

                //Gradients now live in each layer's W_node and b_node after backprop()
                vector<Mat> weight_results(layers.size());
                vector<Mat> bias_results(layers.size());
                for (int i = 0; i < (int)layers.size(); i++) {
                    weight_results[i] = layers[i].W_node->gradient;
                    bias_results[i]   = layers[i].b_node->gradient;
                }

                optimizer->update(layers, weight_results, bias_results);
            }
        }
    }

    double test_accuracy(Mat x_test, Mat y_test) {
        feedforward(0, make_shared<Matrix>(x_test));
        Mat output = layers.back().a;
        int correct = 0;
        for (int col = 0; col < (int)output[0].size(); col++) {
            int pred = 0;
            for (int row = 1; row < (int)output.size(); row++)
                if (output[row][col] > output[pred][col]) pred = row;
            int actual = 0;
            for (int row = 1; row < (int)y_test.size(); row++)
                if (y_test[row][col] > y_test[actual][col]) actual = row;
            if (pred == actual) correct++;
        }
        return (double)correct / output[0].size() * 100.0;
    }
};

int main(int argc, char** argv) {
    ifstream img_file("MNIST handwritten /train-images-idx3-ubyte/train-images-idx3-ubyte", ios::binary);
    uint32_t magic = read_be_uint32(img_file);
    uint32_t num   = read_be_uint32(img_file);
    uint32_t rows  = read_be_uint32(img_file);
    uint32_t cols  = read_be_uint32(img_file);

    vector<uint8_t> raw_images(num * rows * cols);
    img_file.read(reinterpret_cast<char*>(raw_images.data()), raw_images.size());
    img_file.close();

    ifstream lbl_file("MNIST handwritten /train-labels-idx1-ubyte/train-labels-idx1-ubyte", ios::binary);
    uint32_t lbl_magic = read_be_uint32(lbl_file);
    uint32_t lbl_num   = read_be_uint32(lbl_file);

    vector<uint8_t> labels(lbl_num);
    lbl_file.read(reinterpret_cast<char*>(labels.data()), labels.size());
    lbl_file.close();

    Mat vectorized_labels = vectorize_labels(labels, 10);

    int img_dimension = rows * cols;
    Mat image_input_matrix(img_dimension, Vec(num, 0.0));
    for (int column = 0; column < (int)num; column++)
        for (int row = 0; row < img_dimension; row++)
            image_input_matrix[row][column] = raw_images[column * img_dimension + row] / 255.0;

    int num_examples = image_input_matrix[0].size();
    int train_size = (int)(0.666666 * num_examples);

    vector<int> indices(num_examples);
    iota(indices.begin(), indices.end(), 0);
    mt19937 gen(random_device{}());
    shuffle(indices.begin(), indices.end(), gen);

    vector<int> train_indices(indices.begin(), indices.begin() + train_size);
    vector<int> test_indices(indices.begin() + train_size, indices.end());

    Mat x_train(image_input_matrix.size(), Vec(train_size, 0.0));
    Mat x_test(image_input_matrix.size(), Vec(test_indices.size(), 0.0));
    for (int row = 0; row < (int)image_input_matrix.size(); row++) {
        for (int i = 0; i < train_size; i++)
            x_train[row][i] = image_input_matrix[row][train_indices[i]];
        for (int i = 0; i < (int)test_indices.size(); i++)
            x_test[row][i] = image_input_matrix[row][test_indices[i]];
    }

    Mat y_train(vectorized_labels.size(), Vec(train_size, 0.0));
    Mat y_test(vectorized_labels.size(), Vec(test_indices.size(), 0.0));
    for (int row = 0; row < (int)vectorized_labels.size(); row++) {
        for (int i = 0; i < train_size; i++)
            y_train[row][i] = vectorized_labels[row][train_indices[i]];
        for (int i = 0; i < (int)test_indices.size(); i++)
            y_test[row][i] = vectorized_labels[row][test_indices[i]];
    }

    Adam adam(0.001);
    Network MNIST_Network(5, {512, 256, 128, 64, 10}, x_train, FunctionType::RELU, &adam, 0.3);
    MNIST_Network.train_loop(50, y_train, 128);

    //Important to set dropout_rate to 0
    MNIST_Network.dropout_rate = 0;
    double accuracy = MNIST_Network.test_accuracy(x_test, y_test);
    cout << "Test accuracy: " << accuracy << "%" << endl;

    return 0;
}
