from flask import Flask,jsonify
from flask_restplus import Api, fields, Resource
from pathlib import Path
from Gender_Classifier import retrainModel, normalize, name_encoding

import tensorflow as tf
import numpy as np
import csv

app = Flask(__name__)

api = Api(
    app, 
    version='1.X', 
    title='Gender Classifer API',
    description='This Application Programming Interface is used to predict the Gender of a Person given the name of the person')

ns = api.namespace('api')

parser = api.parser()
parser.add_argument(
    'Name', 
    required=True, 
    type= str,
    help='Give the Name of a person like Ajay',
    location='form',
    action='append')

modelParser = api.parser()
modelParser.add_argument(
    'Name', 
    required=True, 
    type= str,
    help='Give the Name of a person like Ajay',
    location='form')
modelParser.add_argument(
    'Gender', 
    required=True, 
    type= str,
    help='M/F',
    location='form')

@ns.route('/classifyGender')
class ClassifiyGender(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        nameList = args['Name']
        resultList = []
        for name in nameList:
            if(name.isalpha()):
                result = self.get_result(name)
                resultList.append(result)
            else:
                return app.response_class(response="Error",status=404)
        
        response = jsonify(resultList)
        response.status_code=200
        return response

    def get_result(self, name):
        model_dir = Path("Gender_Classifier/gender_model.h5")
        model = tf.keras.models.load_model(model_dir)
        nameList = [name]
        prediction = model.predict(np.asarray([np.asarray(name_encoding(normalize(name))) for name in nameList]))
        return {
            'Name':name,
            'Male':(prediction.tolist())[0][0],
            'Female':(prediction.tolist())[0][1]
        }

@ns.route('/retrainModel')
class RetrainModel(Resource):
    @api.doc(parser=modelParser)
    def post(self):
        args = modelParser.parse_args()
        dataset_dir = Path("Gender_Classifier/name_gender.csv")
        with open(dataset_dir,'a') as f:
            writer = csv.writer(f)
            writer.writerow([args['Name'],args['Gender'],1])

        retrainModel()
        return app.response_class(response="Success",status=200)

if __name__ == '__main__':
    app.run(debug=True)
