# imports
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distancepipe = make_pipeline(DistanceTransformer())
        timepipe = make_pipeline(TimeFeaturesEncoder('pickup_datetime'))

        preproc = make_column_transformer((distancepipe, [
            'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
            'dropoff_longitude'
        ]), (timepipe, ['pickup_datetime']))


        pipe = Pipeline([
            ('preproc', preproc),
            ('model',LinearRegression())
        ])
        self.pipe = pipe

    def run(self):
        """set and train the pipeline"""
        self.pipe.fit(self.X,self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse



if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    clean_df = clean_data(df)

    # set X and y
    X = clean_df.drop("fare_amount", axis=1)
    y = clean_df["fare_amount"]

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    # train
    train = Trainer(X,y)
    train.set_pipeline()
    train.run()

    # evaluate
    rmse_score = train.evaluate(X_val,y_val)
    print(rmse_score)
