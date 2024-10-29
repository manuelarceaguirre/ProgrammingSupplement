# api/index.py  
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Michelle Te amo :)!"})

@app.route('/api')
def home():
    return "API is running"
#asda