# This file tests face matching across multiple webcam frames. Instead of relying on one face comparison,
#  it collects up to 10 observations, calculates the average similarity, and displays a possible-match message.
# 14
import cv2
import insightface
import numpy as np

from embedding_utils import cosine_similarity


# --------------------------------
# Configuration
# --------------------------------

DATABASE_PATH = "data/face_database.npz"

PROCESS_EVERY_N_FRAMES = 5

REQUIRED_OBSERVATIONS = 10


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
# Webcam
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


print("\nMulti-frame verification started.")
print("Press Q to quit.")


frame_count = 0

similarity_history = []

current_person = None


# --------------------------------
# Main loop
# --------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    frame_count += 1


    # Process every 5th frame
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        faces = app.get(frame)


        if len(faces) > 0:

            # Select largest face
            face = max(
                faces,
                key=lambda f:
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )


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


            # Best candidate
            best_index = np.argmax(similarities)

            best_similarity = float(
                similarities[best_index]
            )

            best_person_id = str(
                database_person_ids[best_index]
            )


            # --------------------------------
            # Store observation
            # --------------------------------

            similarity_history.append(
                best_similarity
            )

            current_person = best_person_id


            # Keep only required observations
            if len(similarity_history) > REQUIRED_OBSERVATIONS:

                similarity_history.pop(0)


            # --------------------------------
            # Calculate average
            # --------------------------------

            average_similarity = np.mean(
                similarity_history
            )


            print(
                "Observation:",
                len(similarity_history),
                "/",
                REQUIRED_OBSERVATIONS,
                "| Candidate:",
                best_person_id,
                "| Current:",
                round(best_similarity, 3),
                "| Average:",
                round(float(average_similarity), 3)
            )


            # --------------------------------
            # Draw face
            # --------------------------------

            x1, y1, x2, y2 = face.bbox.astype(int)


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Display status
            # --------------------------------

            label = (
                f"{best_person_id} | "
                f"Avg: {average_similarity:.3f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Verification status
            # --------------------------------

            if len(similarity_history) >= REQUIRED_OBSERVATIONS:

                print(
                    "\n10 observations collected."
                )

                print(
                    "Candidate:",
                    current_person
                )

                print(
                    "Average similarity:",
                    round(float(average_similarity), 3)
                )

                print(
                    "Status: POSSIBLE MATCH - "
                    "HUMAN VERIFICATION REQUIRED"
                )

                print("-" * 50)


    # --------------------------------
    # Display camera
    # --------------------------------

    cv2.imshow(
        "Multi-Frame Verification",
        frame
    )


    # --------------------------------
    # Quit
    # --------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# Cleanup
# --------------------------------

cap.release()
cv2.destroyAllWindows()

print("\nMulti-frame verification stopped.")