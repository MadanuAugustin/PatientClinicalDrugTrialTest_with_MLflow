




import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.PatientClinicalDrugTrail.logger_file.logger_obj import logger
from src.PatientClinicalDrugTrail.Exception.custom_exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts//model_trainer//model.joblib'))
        self.preprocessorObj = joblib.load(Path('artifacts//data_transformation//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : 'Age', 1 : 'HB_score', 2 : 'HB_score_3mnths',
                                             3 : 'HB_score_9mnths', 4 : 'Sex', 5 : 'timeOfTreatment',
                                             6 : 'got_Prednisolone', 7 : 'got_Acyclovir'
                                             })
            
            print(data_df)

            transformed_data_df = self.preprocessorObj.transform(data_df)

            transformed_user_input = pd.DataFrame(transformed_data_df)

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_user_input)


            prediction = self.model.predict(transformed_user_input)

            list_output_1  = []

            if prediction[0, 0] == [0.]:
                list_output_1.append('No')
            elif prediction[0, 0] == [1.]:
                list_output_1.append('Yes')

            list_output_2 = []

            if prediction[0, 1] == [0.]:
                list_output_2.append('No')
            elif prediction[0, 1] == [1.]:
                list_output_2.append('Yes')

            return list_output_1, list_output_2
        
        
        except Exception as e:
            raise CustomException(e, sys)

