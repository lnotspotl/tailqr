# tailqr

Implementation of the Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator paper
- Available here: https://arxiv.org/abs/1801.05039
- Check out the original paper: `paper.pdf`
- Check out project report: `report.pdf`
- Most of the code is implemented as part of `impl.ipynb`
- C++ with `openmp` as a dependency was used for simulation speedup: `sample.cpp`

### Compiling approximate gradient descent code

```bash
g++ -fopenmp sample.cpp -o sample -O3
```

### Running approximate gradient descent code
```bash
./sample
```

### Results

![convergence_exact](https://github.com/user-attachments/assets/29f82d24-08de-46fc-9cc0-c2dc3c421177)
![convergence_approx](https://github.com/user-attachments/assets/91170b37-1f18-444a-bc5a-cec4f3aef88b)
