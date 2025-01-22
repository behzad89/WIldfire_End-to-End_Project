from src.componenets.data_transformation_split import DataTranfomationSplit
from src.utils import load_model
from src.logger import logging

# Get the data from source
DATA_PATH='Data/Test.csv'

# Transform & split the data
transformation = DataTranfomationSplit()
df = transformation.data_transformation(DATA_PATH)

# Load the model & prediction
model = load_model('./artifacts/model.pkl')
df['burned_area'] = model.predict(df)
df.to_csv('./artifacts/prediction.csv', index=False)

logging.info('The prediction result was saved.')


