from src.componenets.data_ingestion import DataIngenstion
from src.componenets.data_transformation_split import DataTranfomationSplit
from src.componenets.model_trainer import ModelTrainer

# Get the data from source
obj = DataIngenstion()
data_path = obj.initiate_data_ingestion()

# Transform & split the data
transformation = DataTranfomationSplit()
trainset, testset = transformation.data_split(data_path)

# Train and validation the model
model_trainer = ModelTrainer()
model_trainer.model_train(trainset, testset)


