import cv2
import numpy as np


#Start getting webcam feed
video_capture = cv2.VideoCapture(0)

#Counter to save images
img_counter = 0

#Run the loop
while True:

    # Capture frame-by-frame
    ret1, frame1 = video_capture.read()
    ret2, frame2 = video_capture.read()

    #Set grayscale
    frame1_g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_g = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    frame1_b = cv2.GaussianBlur(frame1_g, (21,21),0)
    frame2_b = cv2.GaussianBlur(frame2_g, (21,21),0)

    diff = cv2.absdiff(frame1_b, frame2_b)

    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]

    final = cv2.dilate(thresh, None, iterations=2)

    masked = cv2.bitwise_and(frame1, frame1, mask=thresh)

    white_pixels = np.sum(thresh)/255

    rows,cols = thresh.shape
    total = rows*cols

    if white_pixels > 0.01*total:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame1, "Motion Detected", (10,50),font,1,(0,0,255),2,cv2.LINE_AA)

    cv2.imshow("Motion", frame1)
    frame1 = frame2
    ret, frame2 = video_capture.read()

    if not ret:
        break

    k = cv2.waitKey(10)

    if k == 2:
        break

    # Exit if ESC Pressed
    if k % 256 == 27:
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
