import cv2
import time
import os
import math
import smtplib
import numpy as np 
from PIL import Image
from pprint import pprint
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

COVID_FACE_MASK_DETECTION_CONFIG = 'cfg/yolov4_train.cfg'
COVID_FACE_MASK_DETECTION_WEIGHTS = 'weights/yolov4_train_3000.weights'
COVID_FACE_MASK_DETECTION_NAMES = 'labels/classes.names'
COVID_FACE_MASK_DETECTION_IMAGES = 'results/'

class CovidFaceMaskDetection():

    def __init__(self, image=None, image_type=None, confidence_threshold=0.3, nms_threshold=0.4):

        self.image = image
        self.image_type = image_type
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold 
        self.net = cv2.dnn.readNetFromDarknet(COVID_FACE_MASK_DETECTION_CONFIG, COVID_FACE_MASK_DETECTION_WEIGHTS)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layers = self.net.getLayerNames()
        self.output_layers = [self.layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def __del__(self):

        del self.image
        del self.image_type
        del self.confidence_threshold
        del self.nms_threshold
        del self.net
        del self.layers
        del self.output_layers

    def load_image(self):

        npimg = np.fromstring(self.image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        #img = cv2.imread(self.image)
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        return img, height, width, channels

    def load_image_streamlit(self, our_image):
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, 1)
        # img = cv2.imread(self.image)
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        return img, height, width, channels

    def detect_objects(self, img):
        
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), \
                        swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return blob, outputs

    def get_box_dimensions(self, outputs, height, width):

        boxes = []
        confs = []
        class_ids = []

        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > self.confidence_threshold:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids

    def draw_labels(self, boxes, confs, class_ids, img, webcam=False):

        with open(COVID_FACE_MASK_DETECTION_NAMES, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        people = 0 
        with_mask_count = 0
        without_mask_count = 0
        detected_labels = []

        indexes = cv2.dnn.NMSBoxes(boxes, confs, self.confidence_threshold, self.nms_threshold)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence = str(round(confs[i]*100,2))
                people += 1 

                if x < 0:
                    x = 0
                elif y < 0:
                    y = 0

                label = str(classes[class_ids[i]])
                detected_labels.append({'bounding_box_coordinates':{'x_top':x,'y_top':y,'width':w,'height':h},\
                                        'label':label, 'confidence': confidence})
              
                color = (0, 255, 0) if label == "mask" else (0, 0, 255)

                if label == "mask":
                    with_mask_count += 1
                    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 1)
                    cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                else:
                    without_mask_count += 1
                    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 1)
                    cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)

        detected_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(COVID_FACE_MASK_DETECTION_IMAGES + "detected_image.jpg", detected_image)

        if without_mask_count != 0:
            self.send_mail_to_admin(without_mask_count)

        if webcam:
            cv2.imshow("Real-time Covid-19 Face Mask Detection using YOLOv4", img)

        response = {'results':detected_labels, 'total_people_count':people,'with_mask_count':with_mask_count,\
                    'without_mask_count':without_mask_count,'status':'ok','error': False, 'algorithm':'YOLOv4'}
       
        return img, response 
       
    def start_webcam(self):
        
        cap = cv2.VideoCapture(0)
        return cap

    def detect_mask_in_webcam(self):

        cap = self.start_webcam()
        while True:

            pprint("Real-time Covid-19 Face Mask Detector is Live..")
            _, frame = cap.read()
            height, width, channels = frame.shape
            blob, outputs = self.detect_objects(frame)
            boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
            detected_image, response = self.draw_labels(boxes, confs, class_ids, frame, webcam=True)
            pprint("------------------------------------------------------")
            print("Real-time Logs: ", response)
            pprint("------------------------------------------------------")

            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()

    def detect_mask_in_image(self): 
    
        start_time = time.time()
        image, height, width, channels = self.load_image()
        blob, outputs = self.detect_objects(image)
        boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)    
        detected_image, response = self.draw_labels(boxes, confs, class_ids, image)
        end_time = time.time() - start_time
        response['inference_time_seconds'] = str(round(end_time,2))
        return response

    def detect_mask_in_image_streamlit(self, our_image, confidence_threshold, nms_threshold): 
    
        start_time = time.time()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        image, height, width, channels = self.load_image_streamlit(our_image)
        blob, outputs = self.detect_objects(image)
        boxes, confs, class_ids = self.get_box_dimensions(outputs, height, width)
        detected_image, response = self.draw_labels(boxes, confs, class_ids, image)
        end_time = time.time() - start_time
        response['inference_time_seconds'] = str(round(end_time,2))
        return detected_image, response

    def send_mail_to_admin(self, without_mask_count):
        
        admin_mail_id = 'replace_admin_email_id'
        system_mail_id = 'replace_system_email_id'
        system_mail_id_password = 'replace_system_email_id_password'
        attachments = [COVID_FACE_MASK_DETECTION_IMAGES + 'detected_image.jpg']

        mail_content = '''Hi Admin,

        Please take the necessary action as {without_mask_count} people without masks are detected.

        Request you to check the attachments too. 

        Hope you will take the action soon.

        This is an auto-generated system mail. Please do not reply!.

        Thank You!
        '''.format(without_mask_count=without_mask_count)

        #attachments.append(EMBS_RESULT_IMAGE_PATH)
        message = MIMEMultipart('alternative')
        message['From'] = system_mail_id
        message['To'] = admin_mail_id
        message['Subject'] = 'Covid-19 Face Mask Detection System Alert!'
        message.attach(MIMEText(mail_content, 'plain'))

        if len('attachments') > 0: 
            for filename in attachments:
                attach_file = open(filename, 'rb') 
                payload = MIMEBase('application', 'octet-stream')
                payload.set_payload((attach_file).read())
                encoders.encode_base64(payload) 
                payload.add_header('Content-Disposition', 'attachment; filename="%s"' %\
                                 os.path.basename(filename))
                message.attach(payload)

        session = smtplib.SMTP('smtp.gmail.com', 587) 
        session.starttls() 
        session.login(system_mail_id, system_mail_id_password) 
        text = message.as_string()
        session.sendmail(system_mail_id, admin_mail_id, text)
        session.quit()
        pprint('Alert Mail Sent to Real-time Covid-19 Face Mask Detector System Admin')
        pprint("---------------------------------------------------")

if __name__ == "__main__":

    obj = CovidFaceMaskDetection('input_images/pic2.jpg','jpg')
    obj.detect_mask_in_webcam()
    #pprint(response)
    #obj.detect_mask_in_image()