from distutils.log import Log
import numpy as np
import 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

random_seed = 50

X = np.load("/home/luke/code/molecular-magic/auto_bandwidth_features/features.npy")
y = np.load("/home/luke/code/molecular-magic/auto_bandwidth_features/labels.npy").astype(np.double)

# TO-DO: Incorperate MolE8 train test split logic
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed)

reg = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(mae)