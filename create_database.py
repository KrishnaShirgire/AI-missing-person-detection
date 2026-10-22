# it takes the registered missing-person photographs, generates face embeddings using InsightFace, 
# and saves those embeddings so the system can compare CCTV faces against them later.
# 09
import os
import cv2
import insightface
import numpy as np


# -----------------------------
# Configuration
# -----------------------------

DATASET_PATH = "data/persons"
DATABASE_PATH = "data/face_database.npz"


# -----------------------------
# Load InsightFace
# -----------------------------

app = insightface.app.FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# -----------------------------
# Store database information
# -----------------------------

embeddings = []
person_ids = []


# -----------------------------
# Read person folders
# -----------------------------

for person_id in os.listdir(DATASET_PATH):

    person_folder = os.path.join(
        DATASET_PATH,
        person_id
    )

    if not os.path.isdir(person_folder):
        continue

    print("\nProcessing:", person_id)


    # Read images
    for filename in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            filename
        )

        # Check image extension
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue


        print("  Image:", filename)


        # Read image
        image = cv2.imread(image_path)

        if image is None:
            print("  Could not read image.")
            continue


        # Detect faces
        faces = app.get(image)


        if len(faces) == 0:
            print("  No face detected.")
            continue


        # Use largest detected face
        face = max(
            faces,
            key=lambda f: (
                f.bbox[2] - f.bbox[0]
            ) * (
                f.bbox[3] - f.bbox[1]
            )
        )


        # Get embedding
        embedding = face.embedding


        # Store
        embeddings.append(embedding)
        person_ids.append(person_id)


        print(
            "  Embedding created:",
            len(embedding)
        )


# -----------------------------
# Check database
# -----------------------------

if len(embeddings) == 0:

    print("\nNo embeddings were created.")
    print("Check your images and folders.")

    exit()


# Convert to NumPy arrays
embeddings = np.array(embeddings)
person_ids = np.array(person_ids)


# -----------------------------
# Save database
# -----------------------------

np.savez(
    DATABASE_PATH,
    embeddings=embeddings,
    person_ids=person_ids
)


print("\n" + "=" * 50)
print("FACE DATABASE CREATED")
print("=" * 50)

print("Total embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)
print("Person IDs:", person_ids)

print("\nDatabase saved at:")
print(DATABASE_PATH)