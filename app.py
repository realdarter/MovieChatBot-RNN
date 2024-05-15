from flask import Flask, render_template, request
from main import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_response', methods=['POST'])
def generate_a_response():
    user_input = request.form['user_input']
    _, response = generate_response(str(user_input).strip()) 
    print(response)
    return render_template('response.html', response=response)


if __name__ == '__main__':
    print("done going to application now.")
    app.run(debug=False)
