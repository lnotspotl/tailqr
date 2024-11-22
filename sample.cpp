#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

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
    vector<pair<double, MatrixXd>> C_estimates;
    vector<MatrixXd> sigma_estimates;

    for (int i = 0; i < m; ++i) {
        // Sample random matrix U_i with Frobenius norm r
        MatrixXd U_i = MatrixXd::Random(K.rows(), K.cols());
        U_i *= r / frobeniusNorm(U_i);

        // Perturbed policy K_i = K + U_i
        MatrixXd K_i = K + U_i;

        // Initialize arrays for trajectory
        double totalCost = 0.0;
        vector<VectorXd> states(rollOutLength, VectorXd(d));
        
        // Sample initial state
        VectorXd x = sampleInitialState(d, gen, dist);

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

        C_estimates.emplace_back(C_i, U_i);
        sigma_estimates.push_back(Sigma_i);
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

pair<MatrixXd, MatrixXd> policyGradientEstimationOptimized(
    const MatrixXd &K, int m, int rollOutLength, double r, int d,
    const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
    std::mt19937 &gen, std::normal_distribution<> &dist
) {
    MatrixXd gradientEstimate = MatrixXd::Zero(K.rows(), K.cols());
    MatrixXd sigmaEstimate = MatrixXd::Zero(d, d);

    // Create thread-local matrices for accumulation
    #pragma omp parallel
    {
        MatrixXd localGradient = MatrixXd::Zero(K.rows(), K.cols());
        MatrixXd localSigma = MatrixXd::Zero(d, d);
        

        #pragma omp for nowait
        for (int i = 0; i < m; ++i) {
            MatrixXd U_i = MatrixXd::Random(K.rows(), K.cols()) * r;

            // Create thread-local random number generator
            std::mt19937 local_gen(gen());
            std::normal_distribution<> local_dist = dist;
            VectorXd x = sampleInitialState(d, local_gen, local_dist);

            MatrixXd K_i = K + U_i;
            MatrixXd closedLoopDynamics = A - B * K_i;
            MatrixXd costMatrix = Q + K.transpose() * R * K;

            vector<VectorXd> states(rollOutLength, VectorXd(d));
            states[0] = x;

            double C_i = 0.0;
            for (int t = 1; t < rollOutLength; ++t) {
                C_i += states[t - 1].transpose() * costMatrix * states[t - 1];
                states[t] = closedLoopDynamics * states[t - 1];
            }

            MatrixXd Sigma_i = MatrixXd::Zero(d, d);
            for (const auto &s : states) {
                Sigma_i += s * s.transpose();
            }

            localGradient += C_i * U_i;
            localSigma += Sigma_i;
        }

        // Critical section for accumulating results
        #pragma omp critical
        {
            gradientEstimate += localGradient;
            sigmaEstimate += localSigma;
        }
    }
    gradientEstimate *= (d / (m * r * r ));
    sigmaEstimate /= m;

    return {gradientEstimate, sigmaEstimate};
}

MatrixXd modelFreeNaturalPolicyGradient(
    const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
    MatrixXd K0, int m, int rollOutLength, double r,
    int maxIterations = 1000, double eta = 0.01
) {
    MatrixXd K = K0;
    int d = A.rows();
    
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < maxIterations; ++i) {
        // Get empirical estimates
        auto [gradEstimate, sigmaEstimate] = policyGradientEstimationNaive(
            K, m, rollOutLength, r, d, A, B, Q, R, gen, dist
        );
        
        std::cout << "Gradient estimate:\n" << gradEstimate << "\n---\n";
        
        // Natural policy gradient update
        MatrixXd K_new = K - eta * gradEstimate;
        
        // // Check convergence
        // if ((K - K_new).norm() < 1e-8) {
        //     break;
        // }
        
        K = K_new;
        std::cout << "Current K:\n" << K << std::endl;
    }
    
    return K;
}

int main() {
    int d = 2;
    int m = 300000;
    int rollOutLength = 150;
    double r = 0.001;

    double dt = 0.1;
    MatrixXd K(1, d); K << 1, 1;
    MatrixXd A = (MatrixXd(d, d) << 1, dt, 0, 1).finished();
    MatrixXd B = (MatrixXd(d, 1) << 0.5 * dt * dt, dt).finished();
    MatrixXd Q = MatrixXd::Identity(d, d) * 2;
    MatrixXd R = MatrixXd::Identity(1, 1);

    // std::mt19937 gen(0);
    // std::normal_distribution<> dist(0.0, 1.0);

    // auto start = std::chrono::high_resolution_clock::now();
    // // auto [gradientNaive, sigmaNaive] = policyGradientEstimationNaive(K, m, rollOutLength, r, d, A, B, Q, R, gen, dist);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "Naive Time Taken: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

    // std::mt19937 gen2(42);
    // start = std::chrono::high_resolution_clock::now();
    // auto [gradientOptimized, sigmaOptimized] = policyGradientEstimationOptimized(K, m, rollOutLength, r, d, A, B, Q, R, gen2, dist);
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Optimized Time Taken: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

    // // std::cout << gradientNaive << std::endl << std::endl << std::endl;
    // std::cout << gradientOptimized << std::endl << std::endl << std::endl;

    // // assert((gradientNaive - gradientOptimized).norm() < 1e-6);

    // // Test model-free natural policy gradient
    MatrixXd K0(1, d);
    K0 << 1.0, 1.0;  // Initial K matrix
    
    std::cout << "Starting model-free natural policy gradient optimization...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    MatrixXd Kest = modelFreeNaturalPolicyGradient(
        A, B, Q, R, K0, 
        3000,    // m
        200,     // rollOutLength
        0.1,    // r
        10000,     // maxIterations
        0.0005    // eta
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Optimization Time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";
    std::cout << "Final K:\n" << Kest << std::endl;

    return 0;
}
