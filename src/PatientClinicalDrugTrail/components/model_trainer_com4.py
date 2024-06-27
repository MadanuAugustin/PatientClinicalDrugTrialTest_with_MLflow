
import pandas as pd
import joblib
import numpy as np
import os
from src.PatientClinicalDrugTrail.entity.config_entity import ModelTrainerConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from src.PatientClinicalDrugTrail.logger_file.logger_obj import logger

class ModelTrainer:
    def __init__(self, config : ModelTrainerConfig):
        self.config = config


    def initiate_model_training(self):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)

        target_train_1 = train_data[[self.config.target_column_1]]
        target_train_2 = train_data[[self.config.target_column_2]]

        train_y = pd.DataFrame(np.column_stack([target_train_1, target_train_2]))

        print(train_y.head())

        test_x = test_data.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)

        target_test_1 = test_data[[self.config.target_column_1]]
        target_test_2 = test_data[[self.config.target_column_2]]

        test_y = pd.DataFrame(np.column_stack([target_test_1, target_test_2]))

        print(test_y.head())

        base_classifier = RandomForestClassifier()

        multi_target_classifier = MultiOutputClassifier(base_classifier)

        multi_target_classifier.fit(train_x, train_y)

        joblib.dump(multi_target_classifier, os.path.join(self.config.root_dir, self.config.model_name))

        logger.info(f'---------------Training of the model completed and saved successfully---------------------')


