import json
from flask import Flask
from flask import request
from flask import url_for

app = Flask(__name__)

@app.route('/')
def hello_world():
    resp = {'code': 1}
    resp['result'] = {}
    resp['result']['userId'] = 101
    return json.dumps(resp)

@app.route('/user/<int:user_id>/<string:user_name>/<email>', methods=['GET'])
def process_path_params(user_id, user_name, email):
    resp = {'code': 1}
    resp['result'] = {}
    resp['result']['userId'] = user_id
    resp['result']['userName'] = user_name
    resp['result']['email'] = email
    return json.dumps(resp)

@app.route('/get_url_params/t1', methods=['GET'])
def get_url_params():
    resp = {'code': 1}
    resp['result'] = {}
    resp['result']['url'] = url_for('get_url_params')
    resp['result']['userId'] = request.args['userId']
    # args类型为ImmutableMultiDict
    params = request.args.to_dict()
    resp['result']['params'] = params
    return json.dumps(resp)

@app.route('/get_post_params/t2', methods=['POST'])
def get_post_params():
    resp = {'code': 1}
    resp['result'] = {}
    resp['result']['userName'] = request.form['userName']
    params = request.form.to_dict()
    return json.dumps(json)


app.run(port=5080, debug=True)