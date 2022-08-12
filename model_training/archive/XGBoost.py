from distutils.log import Log
import numpy as np
from sklearn.metrics import mean_absolute_error
from magic.split import stoichiometric_split
# from sklearn.model_selection import train_test_split

import xgboost as xgb

random_seed = 50

# We are focusing on the prediction of free energy (this is the target)
X = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/features.npy")
y = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/labels.npy").astype(np.double)

X_train, X_test, y_train, y_test = stoichiometric_split(
    X, y, random_state=random_seed)

reg = xgb.XGBRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(mae)