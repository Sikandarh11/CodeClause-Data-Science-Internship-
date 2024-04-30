from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_disease', methods=['POST'])
def check_disease():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the image
        image = cv2.imread(filepath)

        # Perform disease detection (replace this with your actual disease detection logic)
        # For simplicity, let's assume we're just resizing the image
        resized_image = cv2.resize(image, (256, 256))

        # Dummy result
        result = "Healthy"  # Replace with actual disease detection result

        return render_template('result.html', filename=filename, result=result)

    else:
        return render_template('index.html', error='Invalid file format. Please upload an image.')

if __name__ == '__main__':
    app.run(debug=True)
