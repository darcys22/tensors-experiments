// https://pub.towardsai.net/deep-learning-from-scratch-in-modern-c-22bb60c18ff3

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
#pragma GCC diagnostic pop

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>


TEST_CASE( "Std plus", "[std]" ) {
    std::vector<double> X {1., 2., 3., 4., 5., 6.};
    std::vector<double> Y {1., 1., 0., 1., 0., 1.};

    auto result = std::inner_product(X.begin(), X.end(), Y.begin(), 0.0);
    std::cout << "Inner product of X and Y is " << result << '\n';
}

TEST_CASE( "Std accumulate and reduce", "[std]" ) {
    std::vector<double> V {1., 2., 3., 4., 5.};
    double sum = std::accumulate(V.begin(), V.end(), 0.0);
    std::cout << "Summation of V is " << sum << '\n';

    double product = std::accumulate(V.begin(), V.end(), 1.0, std::multiplies<double>());
    std::cout << "Productory of V is " << product << '\n';

    double reduction = std::reduce(V.begin(), V.end(), 1.0, std::multiplies<double>());
    std::cout << "Reduction of V is " << reduction << '\n';
}

double square(double x) {return x * x;}

TEST_CASE( "Std transform and for each", "[std]" ) {
    std::vector<double> X {1., 2., 3., 4., 5.};
    std::vector<double> Y(X.size(), 0);

    std::transform(X.begin(), X.end(), Y.begin(), square);
    std::for_each(Y.begin(), Y.end(), [](double y){ std::cout << y << " ";});
    std::cout << "\n";
}

TEST_CASE( "Functional C++", "[std]" ) {
    std::vector<std::function<bool(double, double)>> comparators
    {
        std::less<double>(),
        std::less_equal<double>(),
        std::greater<double>(),
        std::greater_equal<double>()
    };

    double x = 10.;
    double y = 10.;
    auto compare = [&x, &y](const std::function<bool(double, double)> &comparator)
    {
        bool b = comparator(x, y);
        std::cout << (b?"TRUE": "FALSE") << "\n";
    };

    std::for_each(comparators.begin(), comparators.end(), compare);
}

TEST_CASE( "Momentum C++", "[std]" ) {
    using vector = std::vector<double>;
    auto momentum_optimizer = [V = vector()](const vector &gradient) mutable
    {
        if (V.empty()) V.resize(gradient.size(), 0.);
        std::transform(V.begin(), V.end(), gradient.begin(), V.begin(), [](double v, double dx) 
        {
            double beta = 0.7;
            return v = beta * v + dx; 
        });
        return V;
    };

    auto print = [](double d) { std::cout << d << " "; };
    const vector current_grads {1., 0., 1., 1., 0., 1.};
    for (int i = 0; i < 3; i++)
    {
        vector weight_update = momentum_optimizer(current_grads);
        std::for_each(weight_update.begin(), weight_update.end(), print);
        std::cout << "\n";
    }
}

TEST_CASE( "Eigen ", "[eigen]" ) {
    Eigen::MatrixXd A(2, 2);
    A(0, 0) = 2.;
    A(1, 0) = -2.;
    A(0, 1) = 3.;
    A(1, 1) = 1.;

    Eigen::MatrixXd B(2, 3);
    B(0, 0) = 1.;
    B(1, 0) = 1.;
    B(0, 1) = 2.;
    B(1, 1) = 2.;
    B(0, 2) = -1.;
    B(1, 2) = 1.;

    auto C = A * B;

    std::cout << "A:\n" << A <<std::endl;
    std::cout << "B:\n" << B <<std::endl;
    std::cout << "C:\n" << C <<std::endl;
}
