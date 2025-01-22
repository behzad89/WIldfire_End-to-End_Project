import pandas as pd
from dataclasses import dataclass
import os,sys
from src.exception import CustomException
from src.logger import logging



DATA_PATH='Data/Train.csv'

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngenstion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion or component")
        try:
            df = pd.read_csv(DATA_PATH)
            logging.info('Read the data as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Ingenstion of the data is completed!')

            return self.ingestion_config.raw_data_path
        except Exception as e:
            logging.error( CustomException(e,sys))
            
        

from src.componenets.data_transformation_split import DataTranfomationSplit,DataTransformationSplitConfig 
        
if __name__ == '__main__':
    obj = DataIngenstion()
    data_path = obj.initiate_data_ingestion()
    transformation = DataTranfomationSplit()
    transformation.data_transformation_split(data_path)