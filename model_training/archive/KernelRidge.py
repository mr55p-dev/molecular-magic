# Rest in peace kernel ridge regression... https://datascience.stackexchange.com/questions/28754/sklearn-kernelridge-memory-demand
# tldr; Uses too much RAM.

from distutils.log import Log
import numpy as np
from sklearn.metrics import mean_absolute_error
from magic.split import stoichiometric_split
# from sklearn.model_selection import train_test_split

random_seed = 50

# We are focusing on the prediction of free energy (this is the target)
X = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/features.npy")
y = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/labels.npy").astype(np.double)

X_train, X_test, y_train, y_test = stoichiometric_split(
    X, y, random_state=random_seed)

# More information on Kernel Ridge Regression:
# https://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge
from sklearn.kernel_ridge import KernelRidge

# Can also consider GaussianProcessRegressor:
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor

param_grid = {
    "alpha": [0.1, 1, 10],
    "kernel": ["rbf"] #"linear", "polynomial", "rbf", "MLP"
}

for alpha in param_grid["alpha"]:
    for kernel in param_grid["kernel"]:

        reg = KernelRidge(alpha=alpha, kernel=kernel)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        print("alpha: ",alpha)
        print("kernel:", kernel)
        print(mae)