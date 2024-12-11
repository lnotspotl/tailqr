#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <fstream>
using namespace Eigen;
using namespace std;

// Function to sample initial state from a standard normal distribution
VectorXd sampleInitialState(int d, std::mt19937 &gen, std::normal_distribution<> &dist) {
    VectorXd x(d);
    for (int i = 0; i < d; ++i) {
        x(i) = dist(gen);
    }
    return x;
}

// Function to compute Frobenius norm of a matrix
double frobeniusNorm(const MatrixXd &mat) {
    return sqrt((mat.array() * mat.array()).sum());
}

pair<MatrixXd, MatrixXd> policyGradientEstimationNaive(
    const MatrixXd &K, int m, int rollOutLength, double r, int d, 
    const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
    std::mt19937 &gen, std::normal_distribution<> &dist
) {
    vector<pair<double, MatrixXd>> C_estimates(m);
    vector<MatrixXd> sigma_estimates(m);

    // Pre-allocate vectors to avoid race conditions
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        // Create thread-local random number generator to avoid race conditions
        std::mt19937 thread_gen(gen());
        std::normal_distribution<> thread_dist = dist;

        // Sample random matrix U_i with Frobenius norm r
        MatrixXd U_i = MatrixXd::Random(K.rows(), K.cols());
        U_i *= r / frobeniusNorm(U_i);

        // Perturbed policy K_i = K + U_i
        MatrixXd K_i = K + U_i;

        // Initialize arrays for trajectory
        double totalCost = 0.0;
        vector<VectorXd> states(rollOutLength, VectorXd(d));
        
        // Sample initial state from uniform distribution with [-2sqrt(3), 2sqrt(3)]
        VectorXd x = VectorXd::Random(d) * 2 * sqrt(3);

        for (int t = 0; t < rollOutLength; ++t) {
            states[t] = x;
            VectorXd u = -K_i * x;
            totalCost = totalCost + x.transpose() * Q * x + u.transpose() * R * u;
            x = A * x + B * u;
        }

        // Calculate empirical estimates
        double C_i = totalCost;
        MatrixXd Sigma_i = MatrixXd::Zero(d, d);
        for (const auto &s : states) {
            Sigma_i += s * s.transpose();
        }

        // Store results in pre-allocated vectors
        C_estimates[i] = {C_i, U_i};
        sigma_estimates[i] = Sigma_i;
    }

    // Calculate final estimates
    MatrixXd gradientEstimate = MatrixXd::Zero(K.rows(), K.cols());
    for (const auto &[C_i, U_i] : C_estimates) {
        gradientEstimate += (d / (m * r * r)) * C_i * U_i;
    }

    MatrixXd sigmaEstimate = MatrixXd::Zero(d, d);
    for (const auto &Sigma_i : sigma_estimates) {
        sigmaEstimate += Sigma_i;
    }
    sigmaEstimate /= m;

    return {gradientEstimate, sigmaEstimate};
}


MatrixXd modelFreeNaturalPolicyGradient(
    const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
    MatrixXd K0, int m, int rollOutLength, double r, MatrixXd K_opt,
    int maxIterations = 1000, double eta = 0.01
) {
    MatrixXd K = K0;
    int d = A.rows();
    
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<> dist(0.0, 1.0);

    vector<double> distances;

    for (int i = 0; i < maxIterations; ++i) {
        // Get empirical estimates
        auto [gradEstimate, sigmaEstimate] = policyGradientEstimationNaive(
            K, m, rollOutLength, r, d, A, B, Q, R, gen, dist
        );
        
        std::cout << i << ". Gradient estimate:\n" << gradEstimate << "\n---\n";
        
        // Natural policy gradient update
        MatrixXd K_new = K - eta * gradEstimate * sigmaEstimate.inverse();

        auto distance = (K - K_opt).norm();
        distances.push_back(distance);
        
        // Check convergence
        if (distance < 1e-5) {
            break;
        }
        
        K = K_new;
        std::cout << "Current K:\n" << K << std::endl;
    }

    // Save distances to file
    std::cout << "Saving distances to file..." << std::endl;
    std::ofstream file("distances_npg.txt");
    for (double d : distances) {
        file << d << "\n";
    }
    file.close();
    return K;
}


MatrixXd modelFreePolicyGradient(
    const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
    MatrixXd K0, int m, int rollOutLength, double r, MatrixXd K_opt,
    int maxIterations = 1000, double eta = 0.01
) {
    MatrixXd K = K0;
    int d = A.rows();
    
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<> dist(0.0, 1.0);

    vector<double> distances;

    for (int i = 0; i < maxIterations; ++i) {
        // Get empirical estimates
        auto [gradEstimate, sigmaEstimate] = policyGradientEstimationNaive(
            K, m, rollOutLength, r, d, A, B, Q, R, gen, dist
        );
        
        std::cout << i << ". Gradient estimate:\n" << gradEstimate << "\n---\n";
        
        // Natural policy gradient update
        MatrixXd K_new = K - eta * gradEstimate;

        auto distance = (K - K_opt).norm();
        distances.push_back(distance);
        
        // Check convergence
        if (distance < 1e-5) {
            break;
        }
        
        K = K_new;
        std::cout << "Current K:\n" << K << std::endl;
    }

    // Save distances to file
    std::cout << "Saving distances to file..." << std::endl;
    std::ofstream file("distances_pg.txt");
    for (double d : distances) {
        file << d << "\n";
    }
    file.close();
    return K;
}

int main() {
    int d = 2;
    int m = 900;
    int rollOutLength = 200;
    double r = 0.7;
    int maxIterations = 3000;
    double eta = 0.009;

    double dt = 0.1;
    MatrixXd K(1, d); K << 1, 1;
    MatrixXd A = (MatrixXd(d, d) << 1, dt, 0, 1).finished();
    MatrixXd B = (MatrixXd(d, 1) << 0.5 * dt * dt, dt).finished();
    MatrixXd Q = MatrixXd::Identity(d, d) * 2;
    MatrixXd R = MatrixXd::Identity(1, 1) * 0.01;

    // Test model-free natural policy gradient
    MatrixXd K0(1, d);
    K0 << 1.0, 1.0;  // Initial K matrix
    
    std::cout << "Starting model-free natural policy gradient optimization...\n";
    auto start = std::chrono::high_resolution_clock::now();

    MatrixXd K_opt(1, d);
    K_opt << 6.977, 7.914;
    
    MatrixXd Kest = modelFreeNaturalPolicyGradient(
        A, B, Q, R, K0, 
        m,    // m
        rollOutLength,     // rollOutLength
        r,    // r
        K_opt,  // K_opt
        maxIterations,     // maxIterations
        eta    // eta
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Optimization Time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";
    std::cout << "Final K:\n" << Kest << std::endl;

    start = std::chrono::high_resolution_clock::now();
    MatrixXd Kest2 = modelFreePolicyGradient(
        A, B, Q, R, K0, 
        m,    // m
        rollOutLength,     // rollOutLength
        r,    // r
        K_opt,  // K_opt
        maxIterations,     // maxIterations
        eta    // eta
    );
    end = std::chrono::high_resolution_clock::now();

    std::cout << "Final K:\n" << Kest2 << std::endl;
    std::cout << "Distance:\n" << (Kest2 - K_opt).norm() << std::endl;
    std::cout << "Time:\n" << std::chrono::duration<double>(end - start).count() << " seconds\n";

    return 0;
}
