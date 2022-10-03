# import the necessary packagesimport VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import subprocess
import imutils
import time
import sys
import cv2
import os
import face_recognition
from tkinter import * 
from send_email import sendEmail
from tkinter import messagebox 
#from face_recognition_ai import face_recogntion_method
t_end = time.time() + 50 * 1
t_noMask_end = time.time() + 5 * 1

def face_recogntion_method():
    #try:
        t_end_faceRec = time.time() + 5 * 1
        #top = Tk()
        #top.withdraw()
        # #video_capture = cv2.VideoCapture(0)
        #video_capture = VideoStream(src=0).start()
        print("[INFO] starting video stream...")
        print("[INFO] Recognising Faces...")

        known_people_images = os.listdir('./known_people')
        print(known_people_images)
        known_people_face_encodings = []
        known_people_names = []
        if len(known_people_images) > 0:
            for person in known_people_images:
                known_people_names.append(person.split('.')[0])
                load_image = face_recognition.load_image_file('./known_people/'+person)
                person_face_encodings = face_recognition.face_encodings(load_image)[0]
                known_people_face_encodings.append(person_face_encodings)

            face_locations = []
            face_encodings = []
            face_names = []
            process_this_frame = True
            name = ''
            while time.time() < t_end_faceRec:
                frame = vs.read()
                #ret, frame = video_capture.read()
                frame = imutils.resize(frame, width=1000)
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                if process_this_frame:
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_people_face_encodings, face_encoding)
                        name = "Unknown"
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_people_names[first_match_index]
                        face_distances = face_recognition.face_distance(known_people_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_people_names[best_match_index]
                        face_names.append(name)
                process_this_frame = not process_this_frame
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 3
                    right *= 5
                    bottom *= 5
                    left *= 3
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (255, 0, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)
                
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if name == "Unknown":
                root = Tk()
                root.overrideredirect(1)
                root.withdraw()
                messagebox.showwarning("warning","Please register in known people to get email")
                root.destroy()
                cv2.destroyAllWindows()
            else:
                root = Tk()
                root.overrideredirect(1)
                root.withdraw()
                messagebox.showwarning("warning",name+" pls wear a mask")
                root.destroy()
                cv2.destroyAllWindows()
                sendEmail(name)


def detect_and_predict_nomask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		# post = "No Mask" if preds[0][1] > preds[0][0] else "Mask"
		# if post == "No Mask":
		# 	#countdown(int(5))
		# 	t_end = time.time() + 1100 * 1
		# 	print("Script changed")
		# 	#spawn_program_and_die(['python3.8', '/home/anirudh/Desktop/2nd_year_project/ai_progect_face_recognition.py'])
		# else:
		# 	print("Mask Detected")


	#return a 2-tuple of the face locations and their corresponding
	#locations
	return (locs, preds)


def framedetails():
	while time.time() < t_noMask_end:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=1000)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_nomask(frame, faceNet, maskNet)
		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2) 
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	#cv2.destroyAllWindows()
	#vs.stop()
	print("Video Stream Stopped")



def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		post = "No Mask" if preds[0][1] > preds[0][0] else "Mask"
		if post == "No Mask":
			#countdown(int(5))
			print("Script changed")
			framedetails()
			face_recogntion_method()
			#spawn_program_and_die(['python3.8', '/home/anirudh/Desktop/2nd_year_project/ai_progect_face_recognition.py'])
		else:
			print("Mask Detected")
			return (locs, preds)




# load our serialized face detector model from disk
prototxtPath = r"./face_detector/deploy.prototxt"
weightsPath = r"./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("./mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()


while time.time() < t_end:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2) 
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
print("Video Stream Stopped")