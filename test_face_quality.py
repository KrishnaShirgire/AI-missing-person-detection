#  We test this independently so we can debug quality filtering without running the entire surveillance system.(for testing)
# 05
import cv2
import insightface

from face_quality import check_face_quality


# Create InsightFace
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

print("Face quality testing started.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Detect faces
    faces = app.get(frame)

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # Check quality
        good_face, blur_score = check_face_quality(
            frame,
            face.bbox
        )

        if good_face:

            label = f"GOOD | Blur: {blur_score:.0f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        else:

            label = f"POOR | Blur: {blur_score:.0f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Face Quality Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()