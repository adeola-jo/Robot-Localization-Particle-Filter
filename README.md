
# Particle Filter Algorithm for Robot Localization

This repository contains the implementation of the Particle Filter algorithm used to localize a two-dimensional robot (Turtlebot) in a given map. This work is part of the lab assignments for the Probabilistic Robotics course at the Universitat de Girona, under the Computer Vision and Robotics Research Institute (VICOROB).

## Objective

The aim of this lab is to implement the particle filter algorithm to localize a robot by representing the posterior belief \( \text{bel}(x_t) \) using a set of random particle samples drawn from a Gaussian distribution.

## Implementation

The Particle Filter algorithm consists of three main steps: prediction, weighting (update), and resampling. These steps are recursively performed to estimate the robot's state.

### 1. Prediction

In the prediction step, the new state of each particle is obtained by adding the odometry reading to the previous state of the particle. This step incorporates Gaussian noise to model the uncertainty in odometry measurement.

$$
\begin{pmatrix} 
x_k \\ 
y_k \\ 
\theta_k 
\end{pmatrix} = 
\begin{pmatrix} 
x_{k-1} \\ 
y_{k-1} \\ 
\theta_{k-1} 
\end{pmatrix} + 
\begin{pmatrix} 
\cos(\theta_{k-1}) & -\sin(\theta_{k-1}) \\ 
\sin(\theta_{k-1}) & \cos(\theta_{k-1}) 
\end{pmatrix} 
\begin{pmatrix} 
\Delta x_k + \text{noise}_x \\ 
\Delta y_k + \text{noise}_y 
\end{pmatrix} + 
\begin{pmatrix} 
0 \\ 
0 \\ 
\Delta \theta_k + \text{noise}_\theta 
\end{pmatrix}
$$

### 2. Weighting

In the weighting step, the robot's sensor measurements are used to update the weights of each particle. The likelihood of each particle's predicted measurement is compared to the actual measurement, and the weights are adjusted accordingly.

$$
w = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

where \( x \) is the measured value (range or angle), \( \mu \) is the expected value (extracted from the given map lines), and \( \sigma \) is the uncertainty of the measurement.

### 3. Resampling

The resampling step addresses particle degeneracy by selecting particles with higher weights more frequently, thus focusing the particle set on the most likely states.

$$
N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w_i)^2}
$$

When \( N_{\text{eff}} \) falls below a threshold, resampling is performed using the systematic resampling algorithm. In systematic resampling, particles are selected based on their weights to form a new set of particles.

## Files

- `particle_filter.py`: Contains the implementation of the Particle Filter algorithm.
- `report.pdf`: The lab report detailing the implementation and discussion of the Particle Filter algorithm.

## Disclaimer

This code depends on certain modules developed by the University of Girona and cannot be made public without permission.

The user cannot run the code. This repository is intended to host the results and demonstrate the implementation of the Particle Filter algorithm. Only the parts of the project developed by the authors are made public.

## Results

Here is a GIF showcasing the localization of the robot using the Particle Filter algorithm:

![Particle Filter Result](result.gif)

## Discussion

One issue encountered during the implementation was particle degeneracy, where one particle took almost all the weight, making other particles' contributions insignificant. This was resolved by correctly implementing the weighting step and ensuring proper normalization of the particle weights.

---

**Authors:**
- [Moses Chuka Ebere](https://github.com/MosesEbere)
- [Joseph Oloruntoba Adeola](https://github.com/adeola-jo)

---

## References

- [Probabilistic Robotics by Sebastian Thrun, Wolfram Burgard, and Dieter Fox](https://www.probabilistic-robotics.org/)
- Lecture slides and notes from Universitat de Girona

For more details, refer to the lab report `report.pdf`.

---

### Contact

For any inquiries, please contact:

- Moses Chuka Ebere: moseschukaebere@gmail.com
- Joseph Oloruntoba Adeola: adeolajosepholorum@gmail.com

---

**License:**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

