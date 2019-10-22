import cv2
import urllib.request

#Retereive the officila files of cascade. Can be modeified to use different classifiers.
#eyeURL = "https://docs.opencv.org/3.4.0/haarcascade_eye.xml"
facURL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
#urllib.request.urlretrieve(eyeURL, "/Users/jtmundi/PycharmProjects/Multi_Face_Detection/haarcascade_eye.xml")
urllib.request.urlretrieve(facURL, "/Users/jtmundi/PycharmProjects/Multi_Face_Detection/haarcascade_frontalface_default.xml")

#Opencv cascade classifier for eyes and face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

#Start getting webcam feed
video_capture = cv2.VideoCapture(0)

#Counter to save images
img_counter = 0

#Run the loop
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #Set grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)

    #scale and detect
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces and eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
        #roi_gray = gray[y:y + h, x:x + w]
        #roi_color = frame[y:y + h, x:x + w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('FaceDetection', frame)

    # Exit if ESC Pressed
    if k % 256 == 27:
        break
    # Take a picture if SPACE pressed
    elif k % 256 == 32:
        img_name = "facedetect_webcam_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
