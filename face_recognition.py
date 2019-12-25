# 1. Load the training data (numpy arrays of all the persons)
		# x-values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using OpenCV
# 3. Extract faces out of it
# 4. Use KNN to find the prediction of face (int)
# 5. Map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import cv2
import os

# ============ KNN Code ===============
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]    
    return pred
# ====================================

# Init Camera
cap = cv2.VideoCapture(0)

# Load haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
dataset_path = './data/'
face_data = []
labels = []

class_id = 0	# labels for the given file
names = {}		# Mapping bw id - name

# Data Preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		# Create labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_label = np.concatenate(labels, axis=0)

# Testing
while True:
	ret,frame = cap.read()
	if ret == False:
		continue
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces) == 0:
		continue
	for (x,y,w,h) in faces:
		# Extract the Region of Interest
		offset = 10
		face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# Predicted label (output)
		output = knn(face_dataset,face_label,face_section.flatten())

		# Display on the screen the name and rectangle around it
		pred_name = names[int(output)]
		cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow('Frame',frame)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
