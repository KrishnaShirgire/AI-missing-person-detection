import os
import numpy as np
from insightface.app import FaceAnalysis


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "face_database.npz"
)


# ---------------------------------------------------------
# Load InsightFace model once
# ---------------------------------------------------------

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ---------------------------------------------------------
# Generate embedding from an image
# ---------------------------------------------------------

def generate_embedding(image_path):

    import cv2

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    faces = face_app.get(image)

    if len(faces) == 0:
        raise ValueError(
            "No face detected in the uploaded image."
        )

    # Select the largest detected face
    face = max(
        faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )
    )

    embedding = face.embedding

    if embedding is None:
        raise ValueError(
            "Could not generate face embedding."
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    return embedding


# ---------------------------------------------------------
# Add person embedding to database
# ---------------------------------------------------------

def add_person_to_database(person_id, image_path):

    new_embedding = generate_embedding(image_path)

    # -----------------------------------------------------
    # Existing database
    # -----------------------------------------------------

    if os.path.exists(DATABASE_FILE):

        data = np.load(
            DATABASE_FILE,
            allow_pickle=True
        )

        embeddings = data["embeddings"]

        person_ids = data["person_ids"]

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        person_ids = np.asarray(
            person_ids,
            dtype=str
        )

    else:

        embeddings = np.empty(
            (0, 512),
            dtype=np.float32
        )

        person_ids = np.empty(
            (0,),
            dtype=str
        )

    # -----------------------------------------------------
    # Add new embedding
    # -----------------------------------------------------

    embeddings = np.vstack(
        [
            embeddings,
            new_embedding.reshape(1, -1)
        ]
    )

    person_ids = np.append(
        person_ids,
        person_id
    )

    # -----------------------------------------------------
    # Save database
    # -----------------------------------------------------

    np.savez(
        DATABASE_FILE,
        embeddings=embeddings,
        person_ids=person_ids
    )

    return {
        "person_id": person_id,
        "embedding_size": int(len(new_embedding)),
        "total_embeddings": int(len(embeddings))
    }