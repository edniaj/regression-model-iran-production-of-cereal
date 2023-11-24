from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pathlib
import os
import utils
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'home is sweet'

@app.route('/test')
def test():
    return 'Hello, World!'

@app.route('/data', methods=['GET'])
def data():
    if request.method == 'GET':                
        dict_send_frontend = {
            'variation_2':{
                'max_r2_score_row': utils.read_csv('variation_2_k_fold.csv'),
                },
            'variation_3':{
                'max_r2_score_row': utils.read_csv('variation_3_k_fold.csv')
            }
        }

        return jsonify(dict_send_frontend)
        

if __name__ == '__main__':
    CONTROLLER_PORT = os.getenv('CONTROLLER_PORT')
    print('Flask server running on port', CONTROLLER_PORT) #fix this later
    app.run(debug=True, port=CONTROLLER_PORT)