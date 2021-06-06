import cv2
import time
import os
import math
import numpy as np 
import streamlit as st
from PIL import Image
from pprint import pprint
from CovidFaceMaskDetection import CovidFaceMaskDetection

st.set_page_config(page_title='Covid-19 Face Mask Detector using YOLOv4', page_icon='??', \
					layout='centered', initial_sidebar_state='expanded')

face_mask_detector = CovidFaceMaskDetection()

def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
	
def face_mask_detection():

	local_css("css/styles.css")
	st.markdown('<h1 align="center">?? Covid-19 Face Mask Detection using YOLOv4</h1>', unsafe_allow_html=True)
	activities = ["Image", "Webcam"]
	st.set_option('deprecation.showfileUploaderEncoding', True)
	st.sidebar.markdown("# Mask Detection on?")
	choice = st.sidebar.selectbox("Choose among the given options:", activities)

	if choice == 'Image':
		st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
		st.markdown("### Upload your image here ?")
		image_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])  

		if image_file is not None:
			our_image = Image.open(image_file)  
			st.markdown('<h2 align="left">Original Image</h2>', unsafe_allow_html=True)
			#image_data = image_file.read()
			#our_image = Image.load_img(image_data)
			#im = our_image.save('input_images/a1.jpg')
			st.image(image_file, caption='', width = 600)
			st.markdown('<h4 align="center">Image uploaded successfully!</h4>', unsafe_allow_html=True)

			st.markdown('<h3 align="left">Confidence Threshold</h3>', unsafe_allow_html=True)
			confidence_threshold = st.slider('Please select Confidence Threshold', 0.0, 1.0, step=0.01)
			st.markdown('<h3 align="left">NMS Threshold</h3>', unsafe_allow_html=True)
			nms_threshold = st.slider('Please select NMS Threshold', 0.0, 1.0, step=0.01)
			
			if st.button('Process'):		
				#detected_img, response = detect_mask_in_image(our_image, confidence_threshold, nms_threshold)
				detected_img, response = face_mask_detector.detect_mask_in_image_streamlit(our_image, \
										confidence_threshold, nms_threshold)
				with st.spinner("Detecting..."):
					time.sleep(1)
				total_people_count = response['total_people_count']
				with_mask_count = response['with_mask_count']
				without_mask_count = response['without_mask_count']

				st.markdown('<h2 align="left">Detected Image</h2>', unsafe_allow_html=True)
				st.image(detected_img, width = 600)

				st.markdown('<h2 align="left">Results</h2>', unsafe_allow_html=True)

				st.info("Total People Count: {}".format(total_people_count))
				st.success("With Mask Count: {}".format(with_mask_count))
				st.error("Without Mask Count: {}".format(without_mask_count))

				if without_mask_count != 0:
					with st.spinner("Alert Mail Sent to Admin!"):
						time.sleep(2)

				st.markdown('<h3 align="left">Inference as JSON</h3>', unsafe_allow_html=True)
				st.json(response)
				
face_mask_detection()