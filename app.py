from flask import Flask, render_template, url_for, request, Response
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)


print('ddddddddddddddddd')
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/getData', methods=['GET', 'POST'])
def getData():
    if request.method == 'POST':
        image = request.form['file']
        question = request.form['question']
        s="yes"
        q="wearing spects"
        return render_template('index.html', inputImage=image, ans="{}, {}".format(s,q), qn=question)

@app.route('/execute', methods=['GET', 'POST'])
def execute():
    os.system('python VQA.py')
if __name__ == "__main__":
    app.run(debug=True)
