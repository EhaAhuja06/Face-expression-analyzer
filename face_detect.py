import cv2

# Step 1: Load the pre-trained face classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Step 2: Start webcam
cam = cv2.VideoCapture(0)

while True:
    # Step 3: Read a frame from the camera
    ret, frame = cam.read()
    if not ret:
        print("Camera not detected")
        break

    # Step 4: Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 5: Detect faces
    faces = face_cascade.detectMultiScale(
        gray,          # input image
        scaleFactor=1.3,  # how much image size is reduced at each scale
        minNeighbors=5,   # how many neighbors each candidate rectangle should have
    )

    # Step 6: Draw rectangles around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Step 7: Display the output frame
    cv2.imshow("Face Detection - Press Q to Quit", frame)

    # Step 8: Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release camera and close window
cam.release()
cv2.destroyAllWindows()
