# Face detector(HOG) and draws landmark on faces detected in video feed from webcam.
# https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
# The mouth can be accessed through points [48, 68].
# The right eyebrow through points [17, 22].
# The left eyebrow through points [22, 27].
# The right eye using [36, 42].
# The left eye with [42, 48].
# The nose using [27, 35].
# The jaw via [0, 17].
# 
# My additions to original script:
# Overlay cartoon dog nose and ear graphics have been added at specific shape points.
# Bark when mouth is open.
# Check to see if the puppy is sleepy using eye aspect ratio.
# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf


from imutils import face_utils
import dlib
import cv2
import numpy as np

# For the pygame assets path.
import os

# For aspect ratio sleepy eyes detection.
from scipy.spatial import distance

# For playing sound.
import pygame
pygame.mixer.init()
pygame.mixer.set_num_channels(8) # If you want more channels, change 8 to a desired number. 8 is the default number of channel.
BARK_CHANNEL = pygame.mixer.Channel(5) # This is the sound channel.
BARK = pygame.mixer.Sound(os.path.join('assets', 'bark.mp3'))
SQUEEZE_TOY = pygame.mixer.Sound(os.path.join('assets', 'squeeze_toy.wav'))

 
# p = our pre-treined model directory.
p = "assets/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


cap = cv2.VideoCapture(0)



# Overlay background image with overlay image.
def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


# EAR snippet below borrowed from https://github.com/misbah4064/drowsinessDetector.
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio



def detect_tired_eyes(gray):
    faces = detector(gray)
    for face in faces:
        face_landmarks = predictor(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            # cv2.line(image,(x,y),(x2,y2),(0,255,0),1) # Draw lines around eye.

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            # cv2.line(image,(x,y),(x2,y2),(0,255,0),1) # Draw lines around eye.

        left_ear = calculate_EAR(leftEye) # Calculate left eye aspect ratio.
        right_ear = calculate_EAR(rightEye) # Calculate right eye aspect ratio.

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if (EAR<0.17 and not BARK_CHANNEL.get_busy()):
            BARK_CHANNEL.play(SQUEEZE_TOY)
            #cv2.putText(image,"Sleepy puppy.. Wanna play?",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)


 
while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # Load image of dog nose.
        dog_nose_filter = cv2.imread("assets/dog-nose-xsmall.png", cv2.IMREAD_UNCHANGED)
        # Overlay at point 34 (index 33) and subtract half of image width/height from coordinates.
        image = overlay_transparent(image, dog_nose_filter, shape[34][0]-50, shape[34][1]-17) 

        # Load image of dog ears.
        dog_left_ear_filter = cv2.imread("assets/dog-ear-left-xsmall.png", cv2.IMREAD_UNCHANGED)
        image = overlay_transparent(image, dog_left_ear_filter, shape[0][0]-75, shape[0][1]-50) 
        dog_right_earfilter = cv2.imread("assets/dog-ear-right-xsmall.png", cv2.IMREAD_UNCHANGED)
        image = overlay_transparent(image, dog_right_earfilter, shape[16][0], shape[16][1]-50) 

        # Bark if mouth is open. Check if upper and lower lip points are apart.
        if (shape[66][1]-shape[62][1] >=  18 and not BARK_CHANNEL.get_busy()):
            BARK_CHANNEL.play(BARK)

        # Draw on our image, all the finded cordinate points (x,y). 
        #for (x, y) in shape:
        #    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # Detect if puppy is tired.
    detect_tired_eyes(gray)

    # Show the image
    cv2.imshow("Detecting facial landmarks and putting on a doggy filter. Try to bark!", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()