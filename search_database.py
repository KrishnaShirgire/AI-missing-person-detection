import cv2
import insightface
import numpy as np

from embedding_utils import cosine_similarity


# --------------------------------
# Configuration
# --------------------------------

DATABASE_PATH = "data/face_database.npz"

# Process one face every N frames
PROCESS_EVERY_N_FRAMES = 5


# --------------------------------
# Load database
# --------------------------------

database = np.load(
    DATABASE_PATH,
    allow_pickle=True
)

database_embeddings = database["embeddings"]
database_person_ids = database["person_ids"]

print("Face database loaded.")
print("Total stored embeddings:", len(database_embeddings))


# --------------------------------
# Load InsightFace
# --------------------------------

app = insightface.app.FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# --------------------------------
# Open webcam
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


print("\nFace database search started.")
print("Press Q to quit.")


frame_count = 0

last_person_id = "Searching..."
last_similarity = 0.0


# --------------------------------
# Main loop
# --------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    frame_count += 1


    # Process only every Nth frame
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        faces = app.get(frame)


        if len(faces) > 0:

            # Use largest detected face
            face = max(
                faces,
                key=lambda f:
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )


            # Get embedding
            query_embedding = face.embedding


            # --------------------------------
            # Compare with database
            # --------------------------------

            similarities = []

            for stored_embedding in database_embeddings:

                similarity = cosine_similarity(
                    query_embedding,
                    stored_embedding
                )

                similarities.append(similarity)


            similarities = np.array(similarities)


            # Find best match
            best_index = np.argmax(similarities)

            last_similarity = similarities[best_index]

            last_person_id = database_person_ids[
                best_index
            ]


            print(
                "Best Match:",
                last_person_id,
                "| Similarity:",
                round(float(last_similarity), 4)
            )


            # Face coordinates
            x1, y1, x2, y2 = face.bbox.astype(int)


            # Draw face box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Display result
            label = (
                f"{last_person_id} | "
                f"{last_similarity:.3f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # Display camera
    cv2.imshow(
        "Missing Person Database Search",
        frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# Cleanup
# --------------------------------

cap.release()
cv2.destroyAllWindows()

print("\nSearch stopped.")