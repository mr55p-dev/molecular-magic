from distutils.log import Log
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from magic.split import stoichiometric_split

random_seed = 50

X = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/features.npy")
y = np.load("/home/luke/code/molecular-magic/autoband_badh_freeeng/labels.npy").astype(np.double)

# TO-DO: Incorperate MolE8 train test split logic
X_train, X_test, y_train, y_test = stoichiometric_split(
    X, y, random_state=random_seed)

reg = RidgeCV()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(mae) # Achieves a MAE of 4.336 (using stoichiometric split)

# Achieves a MAE of 3.381 (using new features)