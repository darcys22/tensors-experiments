// https://medium.com/ai-in-plain-english/deep-learning-from-scratch-in-c-tensor-programming-83bca6930e96

#include <numeric>
#include <algorithm> // std::for_each 
#include <functional> // std::less, std::less_equal, std::greater, std::greater_equal
#include <iostream> // std::cout

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#pragma GCC diagnostic pop

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>


TEST_CASE( "Tensor", "[tensor]" ) {
    Eigen::Tensor<int, 3> my_tensor(2,3,4);
    my_tensor.setConstant(42);

    std::cout << "my_tensor:\n\n" << my_tensor << "\n\n";

    std::cout << "tensor size is " << my_tensor.size() << "\n\n";

    my_tensor.setValues({{{1,2,3,4}, {5,6,7,8}}});
    std::cout << "my_tensor:\n\n" << my_tensor << "\n\n";

    Eigen::Tensor<float, 2> kernel(3,3);
    kernel.setRandom();
    std::cout << "kernel:\n\n" << kernel << "\n\n";

}

TEST_CASE( "TensorMap", "[tensor]" ) {
    std::vector<float> storage(4*3);
    std::iota(storage.begin(), storage.end(), 1.);

    for (float v : storage) std::cout << v << ',';
    std::cout << "\n\n";

    Eigen::TensorMap<Eigen::Tensor<float, 2>> my_tensor_view(storage.data(), 4, 3);

    std::cout << "my_tensor_view before update:\n\n" << my_tensor_view << "\n\n";

    // updating the vector
    storage[4] = -1.;

    std::cout << "my_tensor_view after update:\n\n" << my_tensor_view << "\n\n";

    // updating the tensor
    my_tensor_view(2,1) = -8;

    std::cout << "my_tensor_view after update:\n\n" << my_tensor_view << "\n\n";
    std::cout << "vector after two updates:\n\n";
    for (float v : storage) std::cout << v << ',';
    std::cout << "\n\n";
}

TEST_CASE( "Unary and Binary", "[tensor]" ) {
    Eigen::Tensor<float,2> A(2,3), B(2,3);
    A.setRandom();
    B.setRandom();

    Eigen::Tensor<float, 2> C = 2.f*A + B.exp();

    std::cout << "A is\n\n" << A << "\n\n";
    std::cout << "B is\n\n" << B << "\n\n";
    std::cout << "C is\n\n" << C << "\n\n";

    auto cosine = [](float v) {return cos(v);};

    Eigen::Tensor<float,2> D = A.unaryExpr(cosine);
    std::cout << "D is\n\n" << D << "\n\n";

    auto fun = [](float a, float b) {return 2.f*a + b;};
    Eigen::Tensor<float, 2> E = A.binaryExpr(B, fun);
    std::cout << "E is\n\n" << E << "\n\n";
}

// Geometric operations result in tensors with different dimensions and sometimes sizes. Examples are 
// reshape, pad, shuffls, stride, and broadcast.
// Its noteworthy that Eigen Tensor API does not have a transpose operation, we can emulate using shuffle though
auto transpose(const Eigen::Tensor<float, 2> &tensor) {
    Eigen::array<int,2> dims({1,0});
    return tensor.shuffle(dims);
}

TEST_CASE( "transpose", "[tensor]" ) {

    Eigen::Tensor<float, 2> a_tensor(3,4);
    a_tensor.setRandom();

    std::cout << "a_tensor is\n\n" << a_tensor << "\n\n";
    std::cout << "a_tensor transpose is\n\n" << transpose(a_tensor) << "\n\n";
}

// Reductions are a special case of operations that result in a tensor with fewer dimensions. Intuitive cases are sum and maximum
TEST_CASE( "reduction", "[tensor]" ) {

    Eigen::Tensor<float, 3> X(5,2,3);
    X.setRandom();

    std::cout << "X is\n\n" << X << "\n\n";
    std::cout << "X.sum() is\n\n" << X.sum() << "\n\n";
    std::cout << "X.maximum() is\n\n" << X.maximum() << "\n\n";
}

TEST_CASE( "tensor convolutions", "[tensor]" ) {
    Eigen::Tensor<float, 4> input(1,6,6,3);
    input.setRandom();
    Eigen::Tensor<float, 2> kernel(3,3);
    kernel.setRandom();

    Eigen::Tensor<float,4> output(1,4,4,3);

    Eigen::array<int,2> dims({1,2});
    output = input.convolve(kernel, dims);
    std::cout << "input is\n\n" << input << "\n\n";
    std::cout << "kernel is\n\n" << kernel << "\n\n";
    std::cout << "output is\n\n" << output << "\n\n";
}

// Let’s consider the following example where we have two batches of registers, each batch having four registers and each register having three values:
auto softmax(const Eigen::Tensor<float,3> &z)
{
    auto dimensions = z.dimensions();

    int batches = dimensions.at(0);
    int instances_per_batch = dimensions.at(1);
    int instance_length = dimensions.at(2);

    Eigen::array<long int, 1> depth_dim({2});
    auto z_max = z.maximum(depth_dim);

    Eigen::array<long int, 3> reshape_dim({batches, instances_per_batch, 1});
    auto max_reshaped = z_max.reshape(reshape_dim);

    Eigen::array<long int,3> bcast({1,1, instance_length});
    auto max_values = max_reshaped.broadcast(bcast);

    auto diff = z - max_values;
    auto expo = diff.exp();
    auto expo_sums = expo.sum(depth_dim);
    auto sums_reshaped = expo_sums.reshape(reshape_dim);
    auto sums = sums_reshaped.broadcast(bcast);
    auto result = expo / sums;

    return result;
}
TEST_CASE( "softmax with tensors", "[tensor]" ) {
    Eigen::Tensor<float, 3> input(2,4,3);
    input.setValues({
        {{0.1, 1., -2.},{10., 2., 5.},{5., -5., 0.},{2., 3., 2.}},
        {{100., 1000., -500.},{3., 3., 3.},{-1, 1., -1.},{-11., -0.2, -.1}}
    });

    std::cout << "input is\n\n" << input << "\n\n";

    Eigen::Tensor<float, 3> output = softmax(input);
    std::cout << "output is\n\n" << output << "\n\n";

}
