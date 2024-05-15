from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    user_input = request.form['user_input']
    _, response = generate_response(user_input) 
    return render_template('response.html', response=response)


if __name__ == '__main__':
    from main import *
    app.run(debug=True)
