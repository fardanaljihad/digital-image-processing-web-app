import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import random

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/quiz")
@nocache
def quiz():
    return render_template('quiz.html')


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/normal", methods=["POST"])
@nocache
def normal():
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zoomin", methods=["POST"])
@nocache
def zoomin():
    image_processing.zoomin()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zoomout", methods=["POST"])
@nocache
def zoomout():
    image_processing.zoomout()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_left", methods=["POST"])
@nocache
def move_left():
    image_processing.move_left()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_right", methods=["POST"])
@nocache
def move_right():
    image_processing.move_right()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_up", methods=["POST"])
@nocache
def move_up():
    image_processing.move_up()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/move_down", methods=["POST"])
@nocache
def move_down():
    image_processing.move_down()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_addition", methods=["POST"])
@nocache
def brightness_addition():
    image_processing.brightness_addition()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_substraction", methods=["POST"])
@nocache
def brightness_substraction():
    image_processing.brightness_substraction()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_multiplication", methods=["POST"])
@nocache
def brightness_multiplication():
    image_processing.brightness_multiplication()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/brightness_division", methods=["POST"])
@nocache
def brightness_division():
    image_processing.brightness_division()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_equalizer", methods=["POST"])
@nocache
def histogram_equalizer():
    image_processing.histogram_equalizer()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/edge_detection", methods=["POST"])
@nocache
def edge_detection():
    image_processing.edge_detection()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/blur", methods=["POST"])
@nocache
def blur():
    image_processing.blur()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/sharpening", methods=["POST"])
@nocache
def sharpening():
    image_processing.sharpening()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("histogram.html", file_paths=["img/grey_histogram.jpg"])
    else:
        return render_template("histogram.html", file_paths=["img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"])


@app.route("/thresholding", methods=["POST"])
@nocache
def thresholding():
    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    image_processing.threshold(lower_thres, upper_thres)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


# Pertemuan 6
@app.route("/identity", methods=["POST"])
@nocache
def identityKernel():
    image_processing.identityKernel()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/meanfilter", methods=["POST"])
@nocache
def meanFilter():
    image_processing.meanFilter()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/meanfilter-k5", methods=["POST"])
@nocache
def meanFilterK5():
    image_processing.meanFilterK5()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/gaussianfilter-k5", methods=["POST"])
@nocache
def gaussianFilterK5():
    image_processing.gaussianFilterK5()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/gaussianfilter-k25", methods=["POST"])
@nocache
def gaussianFilterK25():
    image_processing.gaussianFilterK25()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/medianfilter-k5", methods=["POST"])
@nocache
def medianFilterK5():
    image_processing.medianFilterK5()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/medianfilter-k25", methods=["POST"])
@nocache
def medianFilterK25():
    image_processing.medianFilterK25()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/sharp", methods=["POST"])
@nocache
def sharp():
    image_processing.sharp()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/bilateralfilter", methods=["POST"])
@nocache
def bilateralFilter():
    image_processing.bilateralFilter()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/zeropadding", methods=["POST"])
@nocache
def zero_padding():
    image_processing.zeroPadding()
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/lowpassfilter", methods=["POST"])
@nocache
def lowpass_filter():
    kernel_size = int(request.form['kernel_size'])
    image_processing.lowPassFilter(kernel_size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/highpassfilter", methods=["POST"])
@nocache
def highpass_filter():
    kernel_size = int(request.form['kernel_size'])
    image_processing.highPassFilter(kernel_size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


@app.route("/bandpassfilter", methods=["POST"])
@nocache
def bandpass_filter():
    kernel_size = int(request.form['kernel_size'])
    image_processing.bandPassFilter(kernel_size)
    return render_template("uploaded.html", file_path="img/img_now.jpg")


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
