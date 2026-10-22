# to check whether InsightFace can detect and analyze a face successfully(for testing)
#  03
import cv2
import insightface

# Create InsightFace application
app = insightface.app.FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

# Prepare the face detection model
app.prepare(ctx_id=0, det_size=(640, 640))

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Face detection started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect faces
    faces = app.get(frame)

    # Draw detected faces
    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show result
    cv2.imshow("InsightFace Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()