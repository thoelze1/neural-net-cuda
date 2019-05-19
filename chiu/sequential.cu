/*
 * In this code, downstream and upstream are relative to data flow direction.  So during the forward pass,
 * downstream is in the forward direction, while during back propagation, downstream is in the backward
 * direction.
 *
 * Also, currently train() and backprop() are currently implemented as separate functions.  I think it is
 * possible, and maybe cleaner, to instead implement them as one function.  On the return, the backprop is
 * done.
 */

#include <cmath>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

using std::size_t;
using Precision = double;

inline double
derivative_error(double n, double d) {
    return std::abs(n - d)/std::max(std::abs(n), std::abs(d));
}

// This holds a sequence of dimensions together in a single type.
template <size_t DP, size_t HP, size_t WP>
struct Dims {
    constexpr static size_t D = DP;
    constexpr static size_t H = HP;
    constexpr static size_t W = WP;
    constexpr static size_t N = D*H*W;
};

template <typename T, size_t D, size_t H, size_t W>
std::ostream &operator<<(std::ostream &os, const T (&a)[D][H][W]) {
    for (size_t h = 0; h < D; h++) {
        if (h > 0) {
            os << "----------" << std::endl;
        }
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                if (j > 0) {
                    os << " ";
                }
                os << std::fixed << std::setprecision(7) << a[h][i][j];
            }
            os << "\n";
        }
    }
    return os;
}

/*
 * Array class:  This is a wrapper around native arrays to get range-checking.
 * It is similar to std::array, but more convenient for multi-dimensional arrays.
 */

// Forward declaration for output operators.
template <typename T, size_t D, size_t... Ds> class Array;

// Output operators for up to 4-D.
template <typename T, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D0> &a) {
    for (size_t i = 0; i < D0; i++) {
        if (i > 0) {
            os << " ";
        }
        os << std::fixed << std::setprecision(7) << a[i];
    }
    os << std::endl;
    return os;
}

template <typename T, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D1, D0> &a) {
    for (size_t i = 0; i < D1; i++) {
        os << std::fixed << std::setprecision(7) << a[i];
    }
    return os;
}

template <typename T, size_t D2, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D2, D1, D0> &a) {
    for (size_t h = 0; h < D2; h++) {
        os << "Matrix " << h << ":" << std::endl;
        os << a[h];
    }
    return os;
}

template <typename T, size_t D3, size_t D2, size_t D1, size_t D0>
std::ostream &
operator<<(std::ostream &os, const Array<T, D3, D2, D1, D0> &a) {
    for (size_t g = 0; g < D3; g++) {
        os << "Tensor " << g << ":" << std::endl;
        os << a[g];
    }
    return os;
}

// General definition of template.
template <typename T, size_t D, size_t... Ds>
class Array {
        friend std::ostream &operator<<<>(std::ostream &, const Array &);
    public:
        Array() = default;
        template <typename U>
        Array(const U &v) {
            *this = v;
        }
        Array<T, Ds...> &operator[](const size_t i) {
            assert(i < D);
            return array[i];
        }
        const Array<T, Ds...> &operator[](const size_t i) const {
            assert(i < D);
            return array[i];
        }
        template <typename... Ts>
        T &operator()(const size_t i, const Ts... rest) {
            return (*this)[i](rest...);
        }
        template <typename... Ts>
        const T &operator()(const size_t i, const Ts... rest) const {
            return (*this)[i](rest...);
        }
        template <typename U>
        Array &operator=(const U &v) {
            std::fill(std::begin(array), std::end(array), v);
            return *this;
        }
        template <typename U>
        Array &operator=(const U (&a)[D]) {
            std::copy(std::begin(a), std::end(a), std::begin(array));
            return *this;
        }
        Array<T, Ds...> *begin() { return &array[0]; }
        Array<T, Ds...> *end() { return &array[D]; }
        const Array<T, Ds...> *begin() const { return &array[0]; }
        const Array<T, Ds...> *end() const { return &array[D]; }
    private:
        Array<T, Ds...> array[D];
};

// Base case.
template <typename T, size_t D>
class Array<T, D> {
        friend std::ostream &operator<<<>(std::ostream &, const Array &);
    public:
        Array() = default;
        template <typename U>
        Array(const U &v) {
            *this = v;
        }
        T &operator[](const size_t i) {
            #ifndef NDEBUG
            if (i >= D) {
                std::cerr << "Index " << i << " beyond end of array of size " << D << "." << std::endl;
                assert(false);
                abort();
            }
            #endif
            return array[i];
        }
        const T&operator[](const size_t i) const {
            #ifndef NDEBUG
            if (i >= D) {
                std::cerr << "Index " << i << " beyond end of array of size " << D << "." << std::endl;
                assert(false);
                abort();
            }
            #endif
            return array[i];
        }
        T &operator()(const size_t i) {
            return (*this)[i];
        }
        const T &operator()(const size_t i) const {
            return (*this)[i];
        }
        template <typename U>
        Array &operator=(const Array<U, D> &a) {
            std::copy(std::begin(a), std::end(a), std::begin(array));
            return *this;
        }
        template <typename U>
        Array &operator=(const U (&a)[D]) {
            std::copy(std::begin(a), std::end(a), std::begin(array));
            return *this;
        }
        template <typename U>
        Array &operator=(const U &v) {
            std::fill(std::begin(array), std::end(array), v);
            return *this;
        }
        T *begin() { return &array[0]; }
        T *end() { return &array[D]; }
        const T *begin() const { return &array[0]; }
        const T *end() const { return &array[D]; }
    private:
        T array[D];
};

// Conversion.
template <typename T1, typename T2> struct ArrayDims;
template <typename T, size_t... Ds>
struct ArrayDims<T, Dims<Ds...>> {
    using type = Array<T, Ds...>;
};

/*
 * Base classes:  These are used as base classes.  HasInputLayer means that it is a layer that accepts input.
 * It does *not* mean that it has an InputLayer (which would be an alternative way to parse the term).
 * HasOutputLayer means that it is a layer that has output.
 */

template <typename T> class HasInputLayer;
template <typename T> class HasOutputLayer;

// Accepts input of the given dimensions.
template <size_t IN_D, size_t IN_H, size_t IN_W>
class HasInputLayer<Dims<IN_D, IN_H, IN_W>> {
    public:
        using InputDims = Dims<IN_D, IN_H, IN_W>;
        using Input = typename ArrayDims<Precision, InputDims>::type;
        HasInputLayer() : previous_layer(nullptr) {
            // Help debugging.
            downstream_deriv = std::numeric_limits<double>::signaling_NaN();
        }
        // Traing modifies and stores the output so that it can be used during backprop.  The last layer will
        // call backprop backward through the layers.
        virtual void train(const int label, const double minibatch_size) = 0;
        virtual void update_weights(const float rate) = 0;
        // Used for checking derivative numerically.
        virtual double loss(const Input &in, const int label) = 0;
        virtual int predict(const Input &) = 0;

    public:
        HasOutputLayer<InputDims> *previous_layer;
    protected:
        // This is passed to the previous layer during backprop.  However, it could be created as simply a
        // temporary array.  The only reason to keep it around is to check whether or not it has been computed
        // correctly.  In other words, it's for debugging.
        Input downstream_deriv;
};

template <typename T> class HasOutputLayer;
template <size_t OUT_D, size_t OUT_H, size_t OUT_W>
class HasOutputLayer<Dims<OUT_D, OUT_H, OUT_W>> {
    public:
        using OutputDims = Dims<OUT_D, OUT_H, OUT_W>;
        using Output = typename ArrayDims<Precision, OutputDims>::type;
        HasOutputLayer() : next_layer(nullptr) {}
        virtual void backprop(const Output &deriv, const double mb_size) = 0;
    public:
        HasInputLayer<OutputDims> *next_layer;
        // Leave public for now so that we can debug easily.
        Output output;
};

/*
 * This layer accepts an input image from MNIST.
 */

template <typename OUT_DIMS>
class InputLayer : public HasOutputLayer<OUT_DIMS> {
    public:
        using OutputIF = HasOutputLayer<OUT_DIMS>;
        using typename OutputIF::Output;
        constexpr static size_t OUT_D = OutputIF::OutputDims::D;
        constexpr static size_t OUT_H = OutputIF::OutputDims::H;
        constexpr static size_t OUT_W = OutputIF::OutputDims::W;
        static_assert(OUT_D == 1, "");
        static_assert(OUT_H >= 1, "");
        static_assert(OUT_W >= 1, "");
    public:
        // This is not virtual, because only layers that have input have train() as part of their interface.
        void train(const float (&image)[OUT_H][OUT_W], const int label, const double mb_size) {
            this->output[0] = image;
            this->next_layer->train(label, mb_size);
        }
        // Because it has output, this function must be defined, but there is no where to backprop to, so
        // there is no need for it to do anything.
        virtual void backprop(const Output &, const double) override { }
        // This is not virtual, because only layers that have input have update_weights() as part of their
        // interface.
        void update_weights(const float rate) {
            this->next_layer->update_weights(rate);
        }
        // This is not virtual, because only layers that have input have predict() as part of their interface.
        int predict(const float (&image)[OUT_H][OUT_W]) {
            Output output;
            output[0] = image;
            return this->next_layer->predict(output);
        }
};

/*
 * FullyConnectedLayer
 */

template <typename IN_DIMS, size_t N_NEURONS>
class FullyConnectedLayer : public HasInputLayer<IN_DIMS>, public HasOutputLayer<Dims<1, 1, N_NEURONS>> {

        using InputIF = HasInputLayer<IN_DIMS>;
        using OutputIF = HasOutputLayer<Dims<1, 1, N_NEURONS>>;
        using typename InputIF::Input;
        using typename OutputIF::Output;
        constexpr static size_t IN_D = InputIF::InputDims::D;
        constexpr static size_t IN_H = InputIF::InputDims::H;
        constexpr static size_t IN_W = InputIF::InputDims::W;
        constexpr static size_t OUT_D = OutputIF::OutputDims::D;
        constexpr static size_t OUT_H = OutputIF::OutputDims::H;
        constexpr static size_t OUT_W = OutputIF::OutputDims::W;
        static_assert(OUT_D == 1, "");
        static_assert(OUT_H == 1, "");

    public:

        FullyConnectedLayer(const std::string &n, const bool relu, const double do_rate, const int seed_seq);
        // This layer has no loss function, so will always call it's forward
        // layer.  If it has no forward layer, that's a bug.
        virtual void train(const int label, const double mb_size) override {

            std::uniform_real_distribution<double> dist(0, 1);
            
            // Fill dropped array with either 0 if dropped, or 1/dropout_rate if not dropped, so that the
            // expected value of the output is constant.
            std::generate(m_current_kept.begin(), m_current_kept.end(),
             [&]() { return dist(m_eng) < m_keep_prob ? 1/m_keep_prob : 0; });
            
            this->forward(this->previous_layer->output, this->m_weight, this->m_bias, this->m_current_kept, this->output);
            this->next_layer->train(label, mb_size);
        }
        virtual void backprop(const Output &full_upstream_deriv, const double mb_size) override;
        virtual void update_weights(const float rate) override;
        void check_weight_derivative(const int label);
        virtual double loss(const Input &in, const int label) override {
            Output temp_output;
            this->forward(in, this->m_weight, this->m_bias, this->m_all_kept, temp_output);
            return this->next_layer->loss(temp_output, label);
        }
        virtual int predict(const Input &in) override {
            Output out;
            this->forward(in, this->m_weight, this->m_bias, this->m_all_kept, out);
            return this->next_layer->predict(out);
        }

    private:

        // Could not make this static bcause it needed the m_relu flag.
        void forward(const Input &input, const Array<Input, N_NEURONS> &weight, const Array<double, N_NEURONS> &bias, const Array<double, N_NEURONS> &dropped, Output &output);

    public:

        const std::string m_name;
        const bool m_relu;
        Array<Input, N_NEURONS> m_weight;
        Array<Input, N_NEURONS> m_weight_deriv;
        Array<double, N_NEURONS> m_bias;
        Array<double, N_NEURONS> m_bias_deriv;
        const double m_keep_prob;
        Array<double, N_NEURONS> m_current_kept;
        const Array<double, N_NEURONS> m_all_kept;

        std::default_random_engine m_eng;
};

template <typename IN_DIMS, size_t N_NEURONS>
FullyConnectedLayer<IN_DIMS, N_NEURONS>::FullyConnectedLayer(const std::string &n, const bool relu, const double do_rate, const int seed_seq)
 : m_name(n), m_relu(relu), m_keep_prob(1 - do_rate), m_all_kept(1), m_eng(7389 + seed_seq) {

    std::normal_distribution<double> init;
    // For each neuron, plane, row, and colum...
    for (auto &n : m_weight) {
        for (auto &p : n) {
            for (auto &r : p) {
                for (auto &c : r) {
                    c = init(m_eng)/sqrt(IN_DIMS::N);
                }
            }
        }
    }

    m_bias = 0;

    m_weight_deriv = 0;
    m_bias_deriv = 0;
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::backprop(const Output &full_upstream_deriv, const double mb_size) {

    auto &upstream_deriv(full_upstream_deriv[0][0]);
    this->downstream_deriv = 0;
    auto &input(this->previous_layer->output);

    for (size_t i = 0; i < N_NEURONS; i++) {
        if (m_current_kept(i) > 0) {
            if (!m_relu || this->output(0, 0, i) > 0) {
                for (size_t in_h = 0; in_h < IN_D; in_h++) {
                    for (size_t in_i = 0; in_i < IN_H; in_i++) {
                        for (size_t in_j = 0; in_j < IN_W; in_j++) {
                            this->downstream_deriv[in_h][in_i][in_j] += m_current_kept(i)*upstream_deriv[i]*m_weight[i][in_h][in_i][in_j];
                            /*
                            fprintf(stderr, "%lu, %lu, %lu: %f\n",
                             in_h, in_i, in_j,
                             this->downstream_deriv[in_h][in_i][in_j]);
                            */
                            // Divide by minibatch size to get the average.
                            m_weight_deriv[i][in_h][in_i][in_j] += (m_current_kept(i)*upstream_deriv[i]*input[in_h][in_i][in_j])/mb_size;
                        }
                    }
                }
                m_bias_deriv(i) += (m_current_kept(i)*upstream_deriv[i])/mb_size;
            }
        }
    }
    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::update_weights(const float rate) {

    for (size_t i = 0; i < N_NEURONS; i++) {
        for (size_t in_h = 0; in_h < IN_D; in_h++) {
            for (size_t in_i = 0; in_i < IN_H; in_i++) {
                for (size_t in_j = 0; in_j < IN_W; in_j++) {
                     m_weight[i](in_h, in_i, in_j) -= rate*m_weight_deriv[i](in_h, in_i, in_j);
                     m_weight_deriv[i](in_h, in_i, in_j) = 0;
                }
            }
        }
        m_bias(i) -= rate*m_bias_deriv(i);
        m_bias_deriv(i) = 0;
    }

    this->next_layer->update_weights(rate);
}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::forward(const Input &input, const Array<Input, N_NEURONS> &weight, const Array<double, N_NEURONS> &bias,
 const Array<double, N_NEURONS> &dropped, Output &output) {
    for (size_t i = 0; i < N_NEURONS; i++) {
        double &out(output[0][0][i]);
        out = 0;
        for (size_t in_h = 0; in_h < IN_D; in_h++) {
            for (size_t in_i = 0; in_i < IN_H; in_i++) {
                for (size_t in_j = 0; in_j < IN_W; in_j++) {
                    out += weight[i][in_h][in_i][in_j]*input[in_h][in_i][in_j];
                }
            }
        }
        out += bias(i);
        if (m_relu) {
            out = std::max(0.0, out);
        }
        assert(dropped(i) == 0 || dropped(i) >= 1);
        out *= dropped(i);
    }
}

/*
 * SoftmaxLayer
 */

template <size_t N>
class SoftmaxLayer : public HasInputLayer<Dims<1, 1, N>>, public HasOutputLayer<Dims<1, 1, N>> {

        using InputIF = HasInputLayer<Dims<1, 1, N>>;
        using OutputIF = HasOutputLayer<Dims<1, 1, N>>;
        using typename InputIF::Input;
        using typename OutputIF::Output;

    public:

        // This layer has no loss function, so will always call it's forward layer.
        // If it has no forward layer, that's a bug.
        virtual void train(const int label, const double mb_size) override {
            forward(this->previous_layer->output, this->output);
            this->next_layer->train(label, mb_size);
        }

        virtual void backprop(const typename OutputIF::Output &full_upstream_deriv, const double mb_size) override;
        virtual void update_weights(const float rate) override {
            // No weights in this layer.
            this->next_layer->update_weights(rate);
        }
        virtual double loss(const Input &in, const int label) override {
            Output temp_output;
            this->forward(in, temp_output);
            return this->next_layer->loss(temp_output, label);
        }
        virtual int predict(const Input &in) override {
            /*
            std::cerr << "Predicting for: " << std::endl;
            for (auto x : in[0][0]) {
                std::cerr << x << std::endl;
            }
            std::cerr << std::endl;
            */
            auto pos = std::max_element(std::begin(in[0][0]), std::end(in[0][0]));
            return std::distance(std::begin(in[0][0]), pos);
        }

    private:

        static void forward(const Input &input, Output &output);
};

template <size_t N>
void
SoftmaxLayer<N>::backprop(const typename OutputIF::Output &full_upstream_deriv, const double mb_size) {

    // Note that we assume that ultimately we are computing the derivative of a scalar with respect to
    // each element of the softmax, so we simply add the derivatives.
    //
    auto &upstream_deriv(full_upstream_deriv[0][0]);
    this->downstream_deriv = 0;
    auto &downstream_deriv(this->downstream_deriv[0][0]);
    auto &output(this->output[0][0]);
    for (size_t j = 0; j < N; j++) {
        downstream_deriv[j] = 0;
        for (size_t i = 0; i < N; i++) {
            if (i == j) {
                downstream_deriv[j] += upstream_deriv[i]*(output[i]*(1 - output[j]));
            } else {
                downstream_deriv[j] += upstream_deriv[i]*(-output[j]*output[i]);
            }
        }
    }
    this->previous_layer->backprop(this->downstream_deriv, mb_size);
}

template <size_t N>
void
SoftmaxLayer<N>::forward(const Input &input, Output &output) {
    // Assume just a 1-D vector.  Note that this is a bit confusing,
    // because in C++, we think of this as just a single row, but
    // mathematically, we like to think of it as a column vector.
    auto &out(output[0][0]);
    auto &in(input[0][0]);
    // D is constant to improve numeric stability.
    const double D = *std::max_element(std::begin(in), std::end(in));
    double sum = 0;
    for (size_t i = 0; i < N; i++) {
        out[i] = exp(in[i] - D);
        sum += out[i];
    }
    for (size_t i = 0; i < N; i++) {
        out[i] = out[i]/sum;
    }
}

/*
 * CrossEntropyLayer
 */

template <size_t N>
class CrossEntropyLayer : public HasInputLayer<Dims<1, 1, N>> {
        using InputIF = HasInputLayer<Dims<1, 1, N>>;
        using typename InputIF::Input;
    public:
        virtual void train(const int label, const double mb_size) override {
            // Note that there is no actual need to calculate the loss at this point.
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wunused-variable"
            double loss = -log(this->previous_layer->output[0][0][label]);
            #pragma GCC diagnostic pop
            // fprintf(stderr, "loss: %f\n", loss);
            Input deriv;
            deriv = 0;
            this->downstream_deriv = 0;
            this->downstream_deriv[0][0][label] = -1/(this->previous_layer->output[0][0][label]);
            this->previous_layer->backprop(this->downstream_deriv, mb_size);
        }
        virtual void update_weights(const float) override {
            // No weights in this layer, and this layer has no output.
        }
        virtual double loss(const Input &in, const int label) override {
            return -std::log(in[0][0][label]);
        }
        virtual int predict(const Input &) override {
            assert(false);
            return -1;
        }
};

void
swap(int &i) {
    // Some of the & are superfluous.
    i =
     (0xff&(i >> 24)) |
     (0xff00&(i >> 8)) |
     (0xff0000&(i << 8)) |
     (0xff000000&(i << 24));
}

int
read_int(int fd) {
    int rv;
    int i;
    rv = read(fd, &i, 4); assert(rv == 4);
    swap(i);
    return i;
}

template <int N>
void
read_mnist_images(const std::string &fn, float (&imgs)[N][28][28]) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n_images = read_int(fd);
    assert(n_images == N);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    for (int i = 0; i < N; i++) {
        unsigned char tmp[28][28];
        rv = read(fd, tmp, 28*28); assert(rv == 28*28);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                // Make go from -1 to 1.
                imgs[i][r][c] = double(tmp[r][c])/127.5 - 1;
            }
        }
    }

    rv = close(fd); assert(rv == 0);
}

template <int N>
void
read_mnist_labels(const std::string &fn, unsigned char (&labels)[N]) {

    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n_labels = read_int(fd);
    assert(n_labels == N);

    rv = read(fd, labels, N); assert(rv == N);

    rv = close(fd); assert(rv == 0);
}

// Two FC layers, with 1024 neurons in first.
int
main() {

    static float training_images[60000][28][28];
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);

    static unsigned char training_labels[60000];
    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59999] == 8);

    static float test_images[10000][28][28];
    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    static unsigned char test_labels[10000];
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    static InputLayer<Dims<1, 28, 28>> il;
    static FullyConnectedLayer<Dims<1, 28, 28>, 1024> dl1("dl1", true, .3, 1);
    static FullyConnectedLayer<Dims<1, 1, 1024>, 10> dl2("dl2", false, 0, 2);
    static SoftmaxLayer<10> sm;
    static CrossEntropyLayer<10> ce;

    il.next_layer = &dl1; dl1.previous_layer = &il;
    dl1.next_layer = &dl2; dl2.previous_layer = &dl1;
    dl2.next_layer = &sm; sm.previous_layer = &dl2;
    sm.next_layer = &ce; ce.previous_layer = &sm;

    std::default_random_engine eng(9815);
    std::uniform_int_distribution<size_t> pick_test(0, 9999);

    for (int e = 0; e < 2; e++) {

        // Create shuffled sequence of training images.
        std::vector<int> training(60000);
        std::iota(training.begin(), training.end(), 0);
        assert(*--training.end() == 59999);
        std::shuffle(training.begin(), training.end(), eng);

        for (int r = 0; r < 600; r++) {

            if (r%100 == 0) {

                // fprintf(stderr, "Begin predict...."); fflush(stderr);
                int correct = 0;
                for (size_t i = 0; i < 10000; i++) {
                    // fprintf(stderr, "Predict: %d for %lu\n", il.predict(training_images[i]), i);
                    size_t ind = pick_test(eng);
                    if (il.predict(test_images[ind]) == test_labels[ind]) {
                        correct++;
                    }
                }
                fprintf(stderr, "Epoch %d: Round %d: accuracy=%f\n", e, r, correct/10000.0);
            }

            for (size_t i = 0; i < 100; i++) {
                il.train(training_images[training.at(100*r + i)], training_labels[training.at(100*r + i)], 100);
            }
            il.update_weights(.002);
        }
    }
}
