
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.PatientClinicalDrugTrail.logger_file.logger_obj import logger
from src.PatientClinicalDrugTrail.Exception.custom_exception import CustomException
from src.PatientClinicalDrugTrail.pipeline.prediction_pipeline import PredictionPipeline



# initializing the flask app

app = Flask(__name__)


# route to display the home page

@app.route('/predict', methods = ['POST', 'GET'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html')
    

    else : 
        try:
            Age = request.form.get('Age')
            HB_score = request.form.get('HB_score')
            HB_score_3mnths = request.form.get('HB_score_3mnths')
            HB_score_9mnths = request.form.get('HB_score_9mnths')
            Sex = request.form.get('Sex')
            timeOfTreatment = request.form.get('timeOfTreatment')
            got_Prednisolone = request.form.get('got_Prednisolone')
            got_Acyclovir = request.form.get('got_Acyclovir')


            data = [Age, HB_score, HB_score_3mnths, HB_score_9mnths, Sex, timeOfTreatment, got_Prednisolone, got_Acyclovir]
            
            logger.info(f'-----------Feteched data successfully from the user--------------')
            

            data = np.array(data).reshape(1, 8)

            data = pd.DataFrame(data)

            print(data)

            obj = PredictionPipeline()

            list_output_1, list_output_2 = obj.predictDatapoint(data)

            return render_template('index.html', results_1 = str(list_output_1), results_2 = str(list_output_2))


        except Exception as e:
            raise CustomException(e, sys)
        



if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True) ## http://127.0.0.1:5000

