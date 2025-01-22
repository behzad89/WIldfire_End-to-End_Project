import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from dataclasses import dataclass
from src.utils import save_object, eveluate_model
from src.exception import CustomException
from src.logger import logging
import os,sys

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    report_file_path = os.path.join('artifacts','model_scores_report.csv')



params={
    "Linear Regression":{},
    
    "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    # 'n_estimators': [8,16,32,64,128,256]
                    'n_estimators': [8,16]
    }
}

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()

    def model_train(self, train_set, test_set):
        try:
            TARGET_COL='burn_area'
            X_train, y_train, X_test, y_test = (train_set.drop([TARGET_COL], axis=1),
                                                train_set[TARGET_COL],
                                                test_set.drop([TARGET_COL], axis=1),
                                                test_set[TARGET_COL])
            
            models={
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_jobs=-1)
            }

            model_report = eveluate_model(X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        models,
                                        params)
            
            df_report = pd.DataFrame.from_dict(model_report, orient='index')

            # Save the DataFrame to a CSV file
            df_report.to_csv(self.model_trainer_config.report_file_path)

            best_test_score = df_report['test_score'].max()
            best_model_name = df_report['test_score'].idxmax()  
            best_model = models[best_model_name]

            if best_test_score<0.1:
                logging.error('No best model found')
                sys.exit(1)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f'The best Model is {best_model_name} with R2_score: {best_test_score}')
            
            return best_test_score

        except Exception as e:
            logging.error(CustomException(e,sys))