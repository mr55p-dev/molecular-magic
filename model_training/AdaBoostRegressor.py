from distutils.log import Log
import numpy as np
from sklearn.metrics import mean_absolute_error
from molmagic.split import stoichiometric_split
# from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor

random_seed = 50

# We are focusing on the prediction of free energy (this is the target)
X = np.load("/home/luke/code/molecular-magic/mole8rep/features.npy")
y = np.load("/home/luke/code/molecular-magic/mole8rep/labels.npy").astype(np.double)

X_train, X_test, y_train, y_test = stoichiometric_split(
    X, y, random_state=random_seed)



reg = AdaBoostRegressor(base_estimator=Ridge(), random_state=random_seed)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(mae) # MAE 7.3