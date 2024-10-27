from flask import Flask, jsonify
import sys

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    print("Hello endpoint called", file=sys.stderr)
    return jsonify({"message": "Hello from Python backend!"})

@app.route('/')
def home():
    print("Root endpoint called", file=sys.stderr)
    return "API is running"

if __name__ == '__main__':
    app.run(debug=True)