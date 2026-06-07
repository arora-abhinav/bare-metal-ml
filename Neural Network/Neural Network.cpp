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
#include "../custom_math.cpp"

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

class ActivationFunctions {
public:
    double sigmoid(double input_val) {
        return 1.0 / (1.0 + exp(-input_val));
    }

    double ReLU(double input_val) {
        return input_val > 0 ? input_val : 0.0;
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
    Mat z;
    Mat a;
    Mat first_moment_weight;
    Mat second_moment_weight;
    Mat first_moment_bias;
    Mat second_moment_bias;
    Mat dropout_mask;

    Layer(int neuron_num, int input_size, FunctionType function_type) {
        this->function_type = function_type;
        this->neuron_num = neuron_num;
        this->input_size = input_size;

        static mt19937 gen(random_device{}());
        normal_distribution<double> gauss(0.0, 0.01);

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

    double execute_function(double input_val) {
        ActivationFunctions functions;
        if (function_type == FunctionType::RELU) return functions.ReLU(input_val);
        if (function_type == FunctionType::SIGMOID) return functions.sigmoid(input_val);
        return 0.0;
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

    Mat hypothesis(Mat linear, bool softmax = false) {
        Mat linear_copy(linear.size(), Vec(linear[0].size(), 0.0));
        for (int row = 0; row < (int)linear.size(); row++)
            for (int col = 0; col < (int)linear[0].size(); col++)
                linear_copy[row][col] = execute_function(linear[row][col]);

        if (!softmax) return linear_copy;

        for (int col = 0; col < (int)linear[0].size(); col++) {
            double exponential_sum = 0.0;
            for (int row = 0; row < (int)linear.size(); row++)
                exponential_sum += exp(linear_copy[row][col]);
            for (int row = 0; row < (int)linear.size(); row++)
                linear_copy[row][col] = exp(linear_copy[row][col]) / exponential_sum;
        }
        return linear_copy;
    }
};

class LossFunctions {
public:
    Vec regularise(Vec vector, double epsilon = 1e-5) {
        Vec copy(vector.size());
        for (int i = 0; i < (int)vector.size(); i++)
            copy[i] = vector[i] + epsilon;
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

    Network(int layer_num, vector<int> neurons_in_layers, Mat initial_input, FunctionType function_type, Optimizer* optimizer, double dropout_rate = 0.0) {
        this->number_of_layers = layer_num;
        this->neurons_in_layers = neurons_in_layers;
        this->initial_input = initial_input;
        this->optimizer = optimizer;
        this->dropout_rate = dropout_rate;

        for (int i = 0; i < number_of_layers; i++) {
            if (i == 0)
                layers.push_back(Layer(neurons_in_layers[i], initial_input.size(), function_type));
            else
                layers.push_back(Layer(neurons_in_layers[i], neurons_in_layers[i - 1], function_type));
        }
    }

    void feedforward(int layer_index, Mat input) {
        if (layer_index >= (int)layers.size()) return;
        Layer& layer = layers[layer_index];
        Mat linear_res = layer.linearize(layer.parameters, input, layer.bias);
        layer.z = linear_res;
        bool is_last = (layer_index == (int)layers.size() - 1);
        Mat output = is_last ? layer.hypothesis(linear_res, true) : layer.hypothesis(linear_res);

        if (dropout_rate > 0.0 && !is_last) {
            static mt19937 rng(random_device{}());
            uniform_real_distribution<double> dist(0.0, 1.0);
            Mat mask(output.size(), Vec(output[0].size(), 0.0));
            for (int row = 0; row < (int)mask.size(); row++)
                for (int col = 0; col < (int)mask[0].size(); col++)
                    mask[row][col] = dist(rng) >= dropout_rate ? 1.0 : 0.0;
            output = element_wise_multiplication(output, mask);
            output = scalar_multiply_matrix(output, 1.0 / (1.0 - dropout_rate));
            layer.dropout_mask = mask;
        }

        layer.a = output;
        feedforward(layer_index + 1, output);
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

    pair<Mat, Mat> last_layer_backprop(Mat labels, Layer& final_layer, Layer* prev_layer = nullptr) {
        Mat prev_act_T = (prev_layer != nullptr)
            ? transpose_matrix(prev_layer->a)
            : transpose_matrix(current_batch);
        Mat term_one = matrix_addition_and_sub(labels, final_layer.a, "sub");
        Mat final_prod = matrix_with_matrix_multiplication(term_one, prev_act_T);
        Mat res = scalar_multiply_matrix(final_prod, -1.0 / labels[0].size());
        Mat product_two = scalar_multiply_matrix(term_one, -1.0 / labels[0].size());
        return {res, product_two};
    }

    pair<Mat, Mat> previous_layer_backprop(Layer& current_layer, Layer& next_layer, Mat previous_product, FunctionType activation, Layer* previous_layer = nullptr) {
        Mat next_param_T = transpose_matrix(next_layer.parameters);
        Mat multiplied = matrix_with_matrix_multiplication(next_param_T, previous_product);
        Mat product_two;
        if (activation == FunctionType::SIGMOID) {
            Mat ones(current_layer.a.size(), Vec(current_layer.a[0].size(), 1.0));
            Mat term = matrix_addition_and_sub(ones, current_layer.a, "sub");
            Mat product_one = element_wise_multiplication(multiplied, current_layer.a);
            product_two = element_wise_multiplication(product_one, term);
        } else {
            product_two = element_wise_multiplication(multiplied, ReLU_derivative(current_layer.z));
        }
        if (dropout_rate > 0.0 && !current_layer.dropout_mask.empty())
            product_two = element_wise_multiplication(product_two, current_layer.dropout_mask);
        Mat prev_act_T = (previous_layer != nullptr)
            ? transpose_matrix(previous_layer->a)
            : transpose_matrix(current_batch);
        Mat res = matrix_with_matrix_multiplication(product_two, prev_act_T);
        return {res, product_two};
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

                // Select batch columns from input and labels
                current_batch = Mat(initial_input.size(), Vec(batch_size, 0.0));
                Mat batch_labels(train_labels.size(), Vec(batch_size, 0.0));
                for (int row = 0; row < (int)initial_input.size(); row++)
                    for (int j = 0; j < batch_size; j++)
                        current_batch[row][j] = initial_input[row][indices[start + j]];
                for (int row = 0; row < (int)train_labels.size(); row++)
                    for (int j = 0; j < batch_size; j++)
                        batch_labels[row][j] = train_labels[row][indices[start + j]];

                feedforward(0, current_batch);
                double current_loss = total_loss(layers.back().a, [&](Vec y_hat, Vec y) {
                    return loss_fn.cross_entropy_loss(y_hat, y);
                }, batch_labels);
                cout << "Epoch " << epoch << ", Batch " << b << ", Loss: " << current_loss << endl;

                vector<Mat> weight_results(layers.size());
                vector<Mat> bias_results(layers.size());

                auto [last_w, backprop_prod] = (layers.size() > 1)
                    ? last_layer_backprop(batch_labels, layers.back(), &layers[layers.size() - 2])
                    : last_layer_backprop(batch_labels, layers.back());

                weight_results.back() = last_w;
                Mat last_bias_grad(backprop_prod.size(), Vec(1, 0.0));
                for (int row = 0; row < (int)backprop_prod.size(); row++) {
                    double s = 0.0;
                    for (double val : backprop_prod[row]) s += val;
                    last_bias_grad[row][0] = s;
                }
                bias_results.back() = last_bias_grad;

                for (int i = (int)layers.size() - 2; i >= 0; i--) {
                    auto [res, new_prod] = (i > 0)
                        ? previous_layer_backprop(layers[i], layers[i + 1], backprop_prod, layers[i].function_type, &layers[i - 1])
                        : previous_layer_backprop(layers[i], layers[i + 1], backprop_prod, layers[i].function_type);
                    backprop_prod = new_prod;
                    Mat bias_grad(backprop_prod.size(), Vec(1, 0.0));
                    for (int row = 0; row < (int)backprop_prod.size(); row++) {
                        double s = 0.0;
                        for (double val : backprop_prod[row]) s += val;
                        bias_grad[row][0] = s;
                    }
                    bias_results[i] = bias_grad;
                    weight_results[i] = res;
                }

                optimizer->update(layers, weight_results, bias_results);
            }
        }
    }

    double test_accuracy(Mat x_test, Mat y_test) {
        feedforward(0, x_test);
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
    Network MNIST_Network(3, {128, 64, 10}, x_train, FunctionType::RELU, &adam, 0.2);
    MNIST_Network.train_loop(30, y_train, 256);

    double accuracy = MNIST_Network.test_accuracy(x_test, y_test);
    cout << "Test accuracy: " << accuracy << "%" << endl;

    return 0;
}
