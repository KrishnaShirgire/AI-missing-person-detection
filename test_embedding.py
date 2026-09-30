import cv2
import insightface
import numpy as np


# Create InsightFace application
app = insightface.app.FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Face embedding test started.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Detect faces
    faces = app.get(frame)

    for face in faces:

        # Get face embedding
        embedding = face.embedding

        print("Face detected")
        print("Embedding shape:", embedding.shape)
        print("Embedding length:", len(embedding))

        # Calculate embedding magnitude
        magnitude = np.linalg.norm(embedding)

        print("Embedding magnitude:", magnitude)
        print("-" * 40)

        # Draw face box
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
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Face Embedding Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()