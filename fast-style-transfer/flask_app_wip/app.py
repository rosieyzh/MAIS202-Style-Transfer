from flask import Flask, render_template, request, make_response, send_file
from functools import wraps, update_wrapper
from PIL import Image
from generator import Generator
import io
import os
import base64

import numpy as np
import torch

#Initialize the predictor object
generator = Generator()
OUTPUT_FOLDER = '/static/img'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# Initializing flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = OUTPUT_FOLDER

@app.route('/', methods=['GET', 'POST']) #defines endpoint for get request
def upload():
    if request.method == 'POST':
        generated = generator.generate(request)
        generated.save('output.jpg')
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        
        return render_template("index.html", style_out =full_filename)
    else:
        return render_template('index.html', style_out = None)

if __name__ == '__main__':
   app.run()
