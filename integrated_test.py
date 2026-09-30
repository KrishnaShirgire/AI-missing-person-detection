import cv2
import numpy as np
import insightface
import time

from collections import defaultdict

from ultralytics import YOLO

from embedding_utils import cosine_similarity
from face_quality import check_face_quality


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "data/face_database.npz"

PROCESS_FACE_EVERY_N_FRAMES = 5

REQUIRED_OBSERVATIONS = 10

TRACK_TIMEOUT = 2.0


# ============================================================
# TRACK VERIFICATION DATA
# ============================================================

tracks = defaultdict(
    lambda: {
        "first_seen": time.monotonic(),
        "last_seen": time.monotonic(),
        "observations": [],
        "status": "COLLECTING"
    }
)


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLOv8...")

yolo_model = YOLO("yolov8n.pt")


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("Loading InsightFace...")

face_app = insightface.app.FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ============================================================
# LOAD FACE DATABASE
# ============================================================

print("Loading face database...")

database = np.load(
    DATABASE_PATH,
    allow_pickle=True
)

database_embeddings = database["embeddings"]
database_person_ids = database["person_ids"]

print(
    "Database embeddings:",
    len(database_embeddings)
)


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam.")
    exit()


print("\nIntegrated tracking verification started.")
print("Press Q to quit.")


frame_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read frame.")
        break


    frame_count += 1

    current_time = time.monotonic()


    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        classes=[0],
        verbose=False
    )


    active_track_ids = set()


    if len(results) > 0:

        result = results[0]


        # ====================================================
        # CHECK TRACK IDs
        # ====================================================

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )


            # =================================================
            # PROCESS EACH PERSON
            # =================================================

            for box, track_id in zip(
                boxes,
                track_ids
            ):

                active_track_ids.add(track_id)


                x1, y1, x2, y2 = box.astype(int)


                # =============================================
                # CREATE / UPDATE TRACK
                # =============================================

                track = tracks[track_id]

                if len(track["observations"]) == 0:

                    track["first_seen"] = current_time

                track["last_seen"] = current_time


                # =============================================
                # TRACK DURATION
                # =============================================

                duration = (
                    current_time
                    - track["first_seen"]
                )


                # =============================================
                # PERSON BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


                # =============================================
                # FACE PROCESSING
                # =============================================

                if (frame_count % PROCESS_FACE_EVERY_N_FRAMES == 0 and len(track["observations"]) < REQUIRED_OBSERVATIONS):

                    # -----------------------------------------
                    # Crop person
                    # -----------------------------------------

                    person_crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2)
                    ]


                    if person_crop.size == 0:
                        continue


                    # -----------------------------------------
                    # Face detection
                    # -----------------------------------------

                    faces = face_app.get(person_crop)


                    if len(faces) == 0:
                        continue


                    # -----------------------------------------
                    # Select largest face
                    # -----------------------------------------

                    face = max(
                        faces,
                        key=lambda f:
                        (f.bbox[2] - f.bbox[0])
                        *
                        (f.bbox[3] - f.bbox[1])
                    )


                    # -----------------------------------------
                    # Face quality
                    # -----------------------------------------

                    good_face, blur_score = check_face_quality(
                        person_crop,
                        face.bbox
                    )


                    if not good_face:

                        cv2.putText(
                            frame,
                            "Poor Face Quality",
                            (
                                x1,
                                min(
                                    y2 + 20,
                                    frame.shape[0] - 10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2
                        )

                        continue


                    # =========================================
                    # FACE EMBEDDING
                    # =========================================

                    query_embedding = face.embedding


                    # =========================================
                    # DATABASE SEARCH
                    # =========================================

                    similarities = []


                    for stored_embedding in database_embeddings:

                        similarity = cosine_similarity(
                            query_embedding,
                            stored_embedding
                        )

                        similarities.append(similarity)


                    similarities = np.array(similarities)


                    best_index = np.argmax(
                        similarities
                    )


                    best_similarity = float(
                        similarities[best_index]
                    )


                    best_person_id = str(
                        database_person_ids[
                            best_index
                        ]
                    )


                    # =========================================
                    # STORE OBSERVATION
                    # =========================================

                    track["observations"].append(
                        {
                            "person_id": best_person_id,
                            "similarity": best_similarity
                        }
                    )


                    # Keep only latest observations

                    if len(track["observations"]) > REQUIRED_OBSERVATIONS:

                        track["observations"] = (
                            track["observations"][
                                -REQUIRED_OBSERVATIONS:
                            ]
                        )


                    # =========================================
                    # PRINT OBSERVATION
                    # =========================================

                    print(
                        f"Track ID: {track_id} | "
                        f"Observation: "
                        f"{len(track['observations'])}/"
                        f"{REQUIRED_OBSERVATIONS} | "
                        f"Candidate: {best_person_id} | "
                        f"Similarity: "
                        f"{best_similarity:.3f} | "
                        f"Blur: {blur_score:.0f}"
                    )


                # =================================================
                # TRACK-LEVEL VERIFICATION
                # =================================================

                observations = track["observations"]


                if len(observations) > 0:

                    candidate_scores = defaultdict(list)


                    for observation in observations:

                        candidate = observation[
                            "person_id"
                        ]

                        similarity = observation[
                            "similarity"
                        ]

                        candidate_scores[
                            candidate
                        ].append(similarity)


                    # ---------------------------------------------
                    # Candidate with most observations
                    # ---------------------------------------------

                    best_candidate = max(
                        candidate_scores,
                        key=lambda candidate:
                        len(
                            candidate_scores[
                                candidate
                            ]
                        )
                    )


                    scores = candidate_scores[
                        best_candidate
                    ]


                    average_similarity = (
                        sum(scores) / len(scores)
                    )


                    # ---------------------------------------------
                    # Status
                    # ---------------------------------------------

                    if len(observations) >= REQUIRED_OBSERVATIONS:

                        track["status"] = (
                                    "POSSIBLE MATCH - "
                                    "HUMAN VERIFICATION REQUIRED"
                                        )

                    else:

                        track["status"] = "COLLECTING"


                    # =============================================
                    # DISPLAY INFORMATION
                    # =============================================

                    label1 = (
                        f"Track {track_id} | "
                        f"{duration:.1f}s"
                    )


                    label2 = (
                        f"{best_candidate} | "
                        f"{average_similarity:.2f}"
                    )


                    label3 = (
                        f"{len(observations)}/"
                        f"{REQUIRED_OBSERVATIONS}"
                    )


                    cv2.putText(
                        frame,
                        label1,
                        (
                            x1,
                            max(y1 - 35, 20)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 0),
                        2
                    )


                    cv2.putText(
                        frame,
                        label2,
                        (
                            x1,
                            max(y1 - 15, 20)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )


                    cv2.putText(
                        frame,
                        label3,
                        (
                            x1,
                            min(
                                y2 + 20,
                                frame.shape[0] - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )


    # ========================================================
    # REMOVE OLD TRACKS
    # ========================================================

    tracks_to_remove = []


    for track_id, track in tracks.items():

        time_since_seen = (
            current_time
            - track["last_seen"]
        )


        if time_since_seen > TRACK_TIMEOUT:

            tracks_to_remove.append(
                track_id
            )


    # ========================================================
    # PRINT FINAL RESULT FOR EXPIRED TRACK
    # ========================================================

    for track_id in tracks_to_remove:

        track = tracks[track_id]

        observations = track["observations"]


        print("\n----------------------------------------")

        print(
            f"Track {track_id} finished."
        )


        if len(observations) > 0:

            candidate_scores = defaultdict(list)


            for observation in observations:

                candidate_scores[
                    observation["person_id"]
                ].append(
                    observation["similarity"]
                )


            best_candidate = max(
                candidate_scores,
                key=lambda candidate:
                len(
                    candidate_scores[candidate]
                )
            )


            scores = candidate_scores[
                best_candidate
            ]


            average_similarity = (
                sum(scores) / len(scores)
            )


            print(
                "Candidate:",
                best_candidate
            )


            print(
                "Observations:",
                len(observations)
            )


            print(
                "Average similarity:",
                round(
                    average_similarity,
                    3
                )
            )


            if len(observations) >= REQUIRED_OBSERVATIONS:

                print(
                    "Status: "
                    "POSSIBLE MATCH - "
                    "HUMAN VERIFICATION REQUIRED"
                )

            else:

                print(
                    "Status: "
                    "INSUFFICIENT OBSERVATIONS"
                )

        else:

            print(
                "No usable face observations."
            )


        print("----------------------------------------\n")


        del tracks[track_id]


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Missing Person Detection",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nIntegrated tracking verification stopped.")