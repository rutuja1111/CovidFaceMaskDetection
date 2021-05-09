from flask import Flask, redirect, url_for, request, make_response, send_file, jsonify
import os
import json
import sys
from CovidFaceMaskDetection import CovidFaceMaskDetection

APP_PORT = '5000'
COVID_FACE_MASK_DETECTION_IMAGES = 'results/'
#APP_PORT = os.environ["APP_PORT"]

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'pong'

@app.route('/detect_mask',methods = ['POST'])
def recognize():

    if request.headers.has_key('File-Type'):
        file_type = request.headers.get('File-Type')
    else:
        file_type = '.jpg'

    if request.headers.has_key('ConficenceThreshold'):
        confidence_threshold = request.headers.get('ConficenceThreshold')
    else:
        confidence_threshold = 0.5

    if request.headers.has_key('NMSThreshold'):
        nms_threshold = request.headers.get('NMSThreshold')
    else:
        nms_threshold = 0.5

    face_mask_detector.image = request.files['file'].read()
    #read image file string data
    #filestr = request.files['file'].read()

    face_mask_detector.image_type = file_type
    face_mask_detector.confidence_threshold = confidence_threshold
    face_mask_detector.nms_threshold = nms_threshold
    response = face_mask_detector.detect_mask_in_image()
    return jsonify(response)

@app.errorhandler(404)
def page_not_found(e):
    return "Page not found"

@app.route('/get/<filename>', methods=["GET"])
def getfile(filename):

    file_path = COVID_FACE_MASK_DETECTION_IMAGES + filename

    if os.path.exists(file_path):
        return make_response(send_file(file_path, attachment_filename = filename, add_etags = False, cache_timeout = 0))
    else:
        return "404"

if __name__ == '__main__':

    try:   
        print("Initializing Objects..")
        face_mask_detector = CovidFaceMaskDetection()
        print("Starting App..")
        app.run(host = "0.0.0.0", port = APP_PORT, debug = True, use_reloader = True)
    except Exception as e:
        print("Caught an Exception: ", e)