import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Test - Press Q to Quit", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
