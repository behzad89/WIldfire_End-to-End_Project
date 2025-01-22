import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from feature_engine.datetime import DatetimeFeatures
from src.utils import lag_generator, save_object


from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os,sys

@dataclass
class DataTransformationSplitConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataTranfomationSplit():
    def __init__(self):
        self.data_transformation_split_config = DataTransformationSplitConfig()

    def get_transformer_obj(self):
        try:
            FTs_To_Extract = ["month", "quarter"]

            pipe = Pipeline([
                            # create datetime features.
                            ('date', DatetimeFeatures(
                                variables="date",
                                features_to_extract=FTs_To_Extract,
                                drop_original=False
                            )),
                            ('lag', FunctionTransformer(lag_generator))
                        ])
            
            return pipe
        
        except Exception as e:
            raise CustomException(e,sys)
   
    def data_transformation(self,data_path:str):
        try:
            df = pd.read_csv(data_path)
            logging.info('Reading the data is completed.')

            df['date'] = pd.to_datetime(df['ID'].str.split('_', expand=True)[1])
            logging.info('Create the Date column from ID')

            df.drop(['climate_swe', 'landcover_3', 'ID'], axis=1, inplace=True)
            logging.info('The columns were dropped!')

            processor = self.get_transformer_obj()
            df_with_date_lag_feats = processor.fit_transform(df)
            logging.info('Temporal & lag features were created!')

            df_with_date_lag_feats.bfill(inplace=True)
            logging.info('NAN rows were Filled.')

            df_with_date_lag_feats.drop(['date'],axis=1,inplace=True)
            logging.info('Dropped the Date column.')
            
            save_object(
                file_path=self.data_transformation_split_config.preprocessor_obj_file_path,
                obj=processor
            )
            logging.info('Transformation was saved.')

            return df_with_date_lag_feats
        
        except Exception as e:
            logging.error(CustomException(e,sys))

    def data_split(self,data_path:str):
        try:
            df = self.data_transformation(data_path)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_transformation_split_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_transformation_split_config.test_data_path, index=False, header=True)
            logging.info('Train-Test data were split.')

            return (train_set, test_set)
        
        except Exception as e:
            logging.error(CustomException(e,sys))