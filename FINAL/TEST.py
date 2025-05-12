from datasets import DATASETS
from baseline import BaselineRegressor

from logger import Logger

logger = Logger(name="test", log_to_file=True)

logger.error("Starting the program")


baseline = BaselineRegressor(model_name="mlp")

dataset = DATASETS[10]
print(dataset.name)
X, y = dataset.preprocess()
print(X.shape, y.shape)
print(dataset.df.head())



X_train, X_test, y_train, y_test = dataset.split()

y_train = y_train.ravel()  
y_test = y_test.ravel()

# baseline.train(X_train, y_train)
# rmse = baseline.evaluate(X_test, y_test, metric="r2")
# print(rmse)
exit()

print(baseline.get_params())

print("*"*100)
baseline.train_and_tune(X_train, y_train)

# baseline.train(X_train, y_train)

rmse = baseline.evaluate(X_test, y_test, metric="r2")
print(rmse)


