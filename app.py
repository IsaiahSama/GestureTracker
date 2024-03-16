import cv2

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, img = webcam.read()

    img = cv2.flip(img, 1)

    cv2.imshow("Testing", img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()