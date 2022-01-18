import Test_data
from flask import Flask,request
app = Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
    return "Test server"


@app.route('/city')
def city():
    filepath = request.args.get('filepath')
   # return "City problem"+filepath
    return Test_data.classified(filepath)
app.run(debug=True, port=5000)