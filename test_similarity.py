import cv2
import insightface

from embedding_utils import cosine_similarity


# Load InsightFace
app = insightface.app.FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


print("Face similarity test started.")
print("Look at the camera.")
print("Press Q to quit.")


reference_embedding = None


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    faces = app.get(frame)


    if len(faces) > 0:

        # Take the first detected face
        face = faces[0]

        current_embedding = face.embedding


        # Store first face as reference
        if reference_embedding is None:

            reference_embedding = current_embedding.copy()

            print("\nReference face stored.")
            print("Embedding length:", len(reference_embedding))


        else:

            similarity = cosine_similarity(
                reference_embedding,
                current_embedding
            )

            print(
                "Similarity:",
                round(similarity, 4)
            )


        # Draw face box
        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "Face Similarity Test",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()