import pickle

from flask import Flask
from flask import request
from flask import jsonify

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
    y_pred = model.predict
    

    result = {
        'approve': bool(y_pred)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port=1234)



