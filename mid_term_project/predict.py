import pickle

from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

model_file = 'mid_term_project\cc_model.pkl'
dv_file = 'mid_term_project\dv.bin'
with open (model_file, 'rb') as f_in:
    model = pickle.load(f_in)
with open (dv_file, 'rb') as f_in:
   dv = pickle.load(f_in) 

app = Flask('approve')

@app.route('/approve', methods=['POST'])

def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    dX = xgb.DMatrix(X)
    y_pred = model.predict(dX)
    approve = y_pred >= 0.5

    result = {
        'approve_probability': float(y_pred),
        'approve': bool(approve)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port=1234)



