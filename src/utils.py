import pandas as pd

from sklearn.metrics import (mean_absolute_error, 
                            root_mean_squared_error, 
                            r2_score)

from src.exception import CustomException
from src.logger import logging
import sys,os,pickle

def lag_generator(df:pd.DataFrame)->pd.DataFrame:
    try:
        climate_columns = df[[col for col in df.columns if col.startswith(("climate_", "lat","lo", "p", "date_m"))]]
        month_avg = climate_columns.groupby(['lat', 'lon','date_month']).mean().reset_index()
        month_avg.index.name = 'index'

        lags = [1,2]
        lag_cols = month_avg.drop('date_month',axis=1).columns
        for l in lags:
            # Shift the timeseries index to get the lagged versions
            df_shift = month_avg[lag_cols].shift(periods=l)
            # Join back to the original dataframe
            month_avg = month_avg.merge(df_shift, on=['lat','lon','index'], how="left", suffixes=("", f"_lag_{l}"))

            landcover_df = df.drop(climate_columns.columns[2:-1],axis=1)
            data = month_avg.merge(landcover_df, on=['lat','lon','date_month'], how="left")

        return data
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def eveluate_model(X_train,
                   y_train,
                   X_test,
                   y_test,
                   models):
    
    try:
        report={}

        for m in range(len(list(models))):
            model=list(models.values())[m]
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train) 
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[m]] = {
                'train_score': train_model_score,
                'test_score': test_model_score
                }

        return report
    except Exception as e:
        logging.error(CustomException(e,sys))