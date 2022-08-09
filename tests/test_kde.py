from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt


mu, sigma = -5, 10 # mean and standard deviation
s = np.random.normal(-5, 1, 1000)
s1 = np.random.normal(0, 1, 1000)
s2 = np.random.normal(5, 1, 1000)
X = np.concatenate([s, s1, s2])

plt.hist(X)

# automatically determind bandwidth?
kde = gaussian_kde(X)

# kde.evaluate(0.1)

lower_bound = min(X)
upper_bound = max(X)
sample_space = np.linspace(lower_bound, upper_bound, 1000)

# Compute the value of the kde at each point in the sample space
sample_values = kde.evaluate(sample_space)

# Find the minima
"""
[0, 1, 2, 3, 4, 5, 6, 7] <= shifted left
    [0, 1, 2, 3, 4, 5, 6, 7] <= original
        [0, 1, 2, 3, 4, 5, 6, 7] <= shifted right

[2, 3, 4, 5, 6, 7] <= truncated left
[1, 2, 3, 4, 5, 6] <= truncated original
[0, 1, 2, 3, 4, 5] <= truncated right

At a minima, original is less than left and right
"""

minima_left = sample_values[1:-1] < sample_values[2:]
minima_right = sample_values[1:-1] < sample_values[:-2]

# Minima are where the sample is smaller than left and right shifted values
# is_maxima = np.logical_and(sample_values > shift_left, sample_values > shift_right)
is_minima = np.logical_and(minima_left, minima_right)

# Compute the values at these points
minima = sample_space[1:-1][is_minima]