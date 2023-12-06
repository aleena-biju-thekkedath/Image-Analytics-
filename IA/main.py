# libraries are imported
from flask import Flask, render_template, request, flash, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np

UPLOAD_FOLDER = 'uploads'             # uploaded images are saved 
PROCESSED_FOLDER = 'static'           # processed images are saved
ALLOWED_EXTENSIONS = {'webp', 'png', 'jpg', 'jpeg'} # file extensions

# creating the flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# for checking valid file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# making home page
@app.route("/", methods=['GET','POST'])
def home():
    if request.method == "POST":
        operation = request.form.get("operation")
        file = request.files['file']

        if file.filename == '':
            flash('No file chosen')
            return redirect(request.url) # redirects to home page
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filename'] = filename
            session['operation'] = operation
        
        # calling all functions
        if operation == "crp":
            return redirect(url_for('crop')) 
        elif operation == "rsz":
            return redirect(url_for('resize'))
        elif operation == "fltr":
            return redirect(url_for('filter'))
        elif operation == "fce":
            return redirect(url_for('face'))
        else:
            flash("Invalid operation selected")
            return redirect(request.url)

    return render_template("index.html")

# the crop function
@app.route("/crop")
def crop():
    filename = session.get('filename')
    operation = session.get('operation')
    print(f"The operation is {operation} and filename is {filename}")
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    if img is not None:
        if operation == "crp":
            img = cv2.resize(img, (3000, 1700))

            def crop(event, x, y, flags, params):
                nonlocal img
                global flag, ix, iy
                if event == cv2.EVENT_LBUTTONDOWN:  # if left mouse button is clicked
                    flag = True
                    ix, iy = x, y
                elif event == cv2.EVENT_LBUTTONUP:  # if left mouse button is released
                    flag = False
                    fx, fy = x, y
                    cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), thickness=1, color=(0, 0, 0)) # draws the rectangle
                    cropped = img[iy:fy, ix:fx]  # crops the image
                    cv2.imwrite(os.path.join("static", filename), cropped)
                    cv2.destroyAllWindows()
                    return redirect(url_for('display_cropped_image', filename=filename))

            cv2.namedWindow(winname="window", flags=cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("window", crop)
            cv2.imshow("window", img)
            cv2.setWindowProperty("window", cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(0)
    return render_template("crop.html", image_filename=filename)

# Resize function
@app.route("/resize", methods=['GET', 'POST'])
def resize():
    if request.method == "POST":
        # Get the input dimensions from the form
        width = int(request.form.get("width"))
        height = int(request.form.get("height"))

        filename = session.get('filename')
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Resize the image
        resized_img = cv2.resize(img, (width, height))
        resized_filename = f"resized_{filename}"
        cv2.imwrite(os.path.join("static", resized_filename), resized_img)
        return render_template("resize.html", resized_filename=resized_filename)

    return render_template("resize.html")

 # filter function 
@app.route("/filter", methods=['GET','POST'])
def filter():
    if request.method == "POST":
        filter_option = request.form.get("filter_option")
        filename = session.get('filename')
        img = cv2.imread("uploads" + "/" + filename)

        # Apply the selected filter option
        if filter_option == "black_white":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_filtered = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            filtered_filename = f"filtered_{filename}"
            cv2.imwrite(os.path.join("static", filtered_filename), img_filtered)

        elif filter_option == "exposure":
            img_filtered = cv2.Canny(img,50,200) # canny edge detection
            filtered_filename = f"filtered_{filename}"
            cv2.imwrite(os.path.join("static", filtered_filename), img_filtered)

        elif filter_option == "contours":
            blurred = cv2.GaussianBlur(img, (3, 3), 0) 
            edged = cv2.Canny(blurred, 10, 100)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finding contours 
            img_filtered = img.copy()
            cv2.drawContours(img_filtered, contours, -1, (0, 255, 0), 2)  # drawing contours on main image
            filtered_filename = f"filtered_{filename}"
            cv2.imwrite(os.path.join("static", filtered_filename), img_filtered)

        elif filter_option == "blur":
            img_filtered =  cv2.GaussianBlur(img, (15,15), 0) # gaussian blur
            filtered_filename = f"filtered_{filename}"
            cv2.imwrite(os.path.join("static", filtered_filename), img_filtered)   

        else:
            flash("Invalid filter option selected")
            return render_template("filter.html")
        
        return render_template("filter.html", filtered_filename=filtered_filename)

    return render_template("filter.html")

#------------------------------------------------------------------
# finding face 
@app.route("/face")
def face():

    filename = session.get('filename')
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    operation = session.get('operation')
    if img is not None:
        if operation == "fce":

            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # haarcascade file
            face_classifier = cv2.CascadeClassifier("C:/Users/Aleena/Downloads/haarcascade_frontalface_default.xml")
            # finding face in the image 
            face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(35,35))

            for (x, y, w, h) in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # final_face = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            face_filename = f"face_{filename}"
            cv2.imwrite(os.path.join("static", face_filename), img)
            return render_template("face.html", face_filename=face_filename)

    return render_template("face.html")
