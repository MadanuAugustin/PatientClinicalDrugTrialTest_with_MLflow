import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.PatientClinicalDrugTrail.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from src.PatientClinicalDrugTrail.logger_file.logger_obj import logger
from src.PatientClinicalDrugTrail.Exception.custom_exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer





class DataTransformation:
    def __init__(self, config : DataTransformationConfig):
        
        self.config = config




    def data_split(self):
        try:

            logger.info(f'-----------Entered data_split method---------------')
        
            data = pd.read_csv(self.config.local_data_file)

            data = data[['Sex', 'Age', 'HB_score', 'timeOfTreatment', 'got_Prednisolone',
                         'got_Acyclovir', 'HB_score_3mnths', 'cured3mnths', 'HB_score_9mnths', 'cured9mnths']]
            
            data['cured3mnths'] = data['cured3mnths'].map({'Yes' : 1, 'No' : 0})

            data['cured9mnths'] = data['cured9mnths'].map({'Yes\r' : 1, 'No\r' : 0})

            train, test = train_test_split(data, test_size=0.2, random_state=42)

            train.to_csv(os.path.join(self.config.train_path, 'train.csv'), index = False, header = True)
        
            test.to_csv(os.path.join(self.config.test_path, 'test.csv'), index = False, header = True)

            logger.info(f'----------------saved train test data in csv format------------------')

            logger.info(f'------------The shape of the train data is {train.shape}')

            logger.info(f'--------------The shape of the test data is {test.shape}')

            logger.info(f'------------completed data splitting----------------------')

        except Exception as e:
            raise CustomException(e, sys)
        

    def preprocessor_fun(self):
        try:

            logger.info(f'---------------Entered preprocessor function------------------')


            numeric_columns = ['Age', 'HB_score', 'HB_score_3mnths', 'HB_score_9mnths']

            categoric_columns = ['Sex', 'timeOfTreatment', 'got_Prednisolone', 'got_Acyclovir']
            
            logger.info(f'----------creating transformer pipelines---------------')

            numeric_pipeline = Pipeline(
                steps=[
                    ('standardscaler', StandardScaler(with_mean=True))
                ]
            )

            categoric_pipeline = Pipeline(
                steps=[
                    ('onehot', OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('numericpipeline', numeric_pipeline, numeric_columns),
                    ('categoricpipeline', categoric_pipeline, categoric_columns)
                ]
            )

            logger.info(f'---------------completed creating transformer pipelines---------------')

            logger.info(f'--------------completed preprocessor function------------------')

            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)
        



    def initiate_data_transformation(self):

        try:

            logger.info(f'------------started initiate_data_transformation method------------')

            train_df = pd.read_csv('artifacts//data_transformation//train.csv')
            
            test_df = pd.read_csv('artifacts//data_transformation//test.csv')

            logger.info(f'----------obtaining the preprocessor obj-----------')

            independent_train_X = train_df.drop(columns = ['cured3mnths','cured9mnths'], axis = 1)
            dependent_train_Y = train_df[['cured3mnths','cured9mnths']]

            independent_test_X = test_df.drop(columns = ['cured3mnths','cured9mnths'], axis = 1)
            dependent_test_Y = test_df[['cured3mnths','cured9mnths']]

            preprocessor_obj = self.preprocessor_fun()

            transformed_train_df = preprocessor_obj.fit_transform(independent_train_X)

            joblib.dump(preprocessor_obj, os.path.join(self.config.root_dir, 'preprocessor_obj.joblib'))

            transformed_test_df = preprocessor_obj.transform(independent_test_X)

            transformed_train_df = pd.DataFrame(np.c_[transformed_train_df, dependent_train_Y])

            transformed_test_df = pd.DataFrame(np.c_[transformed_test_df, dependent_test_Y])

            transformed_train_df.rename(columns={10 : 'cured3mnths', 11 : 'cured9mnths'}, inplace=True)

            transformed_test_df.rename(columns={10 : 'cured3mnths', 11 : 'cured9mnths'}, inplace=True)

            transformed_train_df.to_csv(os.path.join(self.config.root_dir, 'transformed_train_df.csv'), index = False, header = True)

            transformed_test_df.to_csv(os.path.join(self.config.root_dir, 'transformed_test_df.csv'), index = False, header = True)

            logger.info(f'-------------transformed data using preprocessor obj and saved in csv format----------')

            logger.info(f'--------------completed initiate_data_transformation method--------------')

        except Exception as e:
            raise CustomException(e, sys)