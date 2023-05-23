# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

mlflow.set_experiment("IAD_project")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(2022)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(run_name="test_run_2"):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")

"""
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
sk_model = tree.DecisionTreeClassifier()
sk_model = sk_model.fit(iris.data, iris.target)

# Save the model in cloudpickle format
# set path to location for persistence
sk_path_dir_1 = "test_model"
mlflow.sklearn.save_model(
        sk_model, sk_path_dir_1,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

# save the model in pickle format
# set path to location for persistence
sk_path_dir_2 = "test_model"
mlflow.sklearn.save_model(sk_model, sk_path_dir_2,
                          serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
sk_model = tree.DecisionTreeClassifier()
sk_model = sk_model.fit(iris.data, iris.target)
# set the artifact_path to location where experiment artifacts will be saved

#log model params
mlflow.log_param("criterion", sk_model.criterion)
mlflow.log_param("splitter", sk_model.splitter)

# log model
mlflow.sklearn.log_model(sk_model, "sk_models")
"""

'''
import mlflow
import sklearn
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()

if __name__ == '__main__':
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    # Используем только часть фичей
    X = X.iloc[:, :2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        clf = svm.SVC(kernel="rbf", C=1)
        clf.fit(X_train, y_train)
        val_metrics = mlflow.sklearn.eval_and_log_metrics(clf, X_test, y_test, prefix="val_")
'''