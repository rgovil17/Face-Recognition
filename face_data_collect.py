# Captures images from your webcam video stream
# Extracts all faces from the image frame (using haarcascade)
# Stores the face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image (grayscale, to save memory) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# Init camera
cap = cv2.VideoCapture(0)

# Load haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the person: ")

while True:
	ret,frame = cap.read()
	if ret == False:
		continue
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area)
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		# Extract (crop out the required face) : Region of Interest
		offset = 10
		face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# Store every 10th face
		skip += 1
		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow('Frame',frame)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert face list into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save the data into the file system
path = dataset_path + file_name + '.npy'
np.save(path,face_data)
print('Data successfully saved at ' + path)

cap.release()
cv2.destroyAllWindows()
