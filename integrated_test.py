# It contains real AI-processing logic, but the code you shared runs as a standalone webcam test rather than connecting to your complete dashboard application.
# 15
import cv2
import numpy as np
import insightface
import time

from collections import defaultdict, Counter
from ultralytics import YOLO

from embedding_utils import cosine_similarity
from face_quality import check_face_quality


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "data/face_database.npz"

# Process face recognition every N frames
PROCESS_FACE_EVERY_N_FRAMES = 5

# Number of usable face observations required
REQUIRED_OBSERVATIONS = 10

# Time after which a missing track is considered finished
TRACK_TIMEOUT = 2.0

# ------------------------------------------------------------
# IMPORTANT:
# This is an INITIAL calibration threshold.
# It must be validated with more genuine and unknown samples.
# It is NOT a final universal face-recognition threshold.
# ------------------------------------------------------------
INITIAL_MATCH_THRESHOLD = 0.60


# ============================================================
# TRACK STORAGE
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
# HELPER FUNCTION
# ============================================================

def calculate_track_statistics(observations):
    """
    Calculate statistics for all observations belonging
    to one Track ID.
    """

    if not observations:
        return None

    # Extract candidate IDs
    person_ids = [
        observation["person_id"]
        for observation in observations
    ]

    # Extract similarity scores
    similarities = [
        observation["similarity"]
        for observation in observations
    ]

    # Count how many times each candidate appeared
    candidate_counts = Counter(person_ids)

    # Candidate with the highest number of observations
    candidate, candidate_count = candidate_counts.most_common(1)[0]

    # Similarities belonging to selected candidate
    candidate_similarities = [
        observation["similarity"]
        for observation in observations
        if observation["person_id"] == candidate
    ]

    # Statistics
    average_similarity = float(
        np.mean(candidate_similarities)
    )

    minimum_similarity = float(
        np.min(candidate_similarities)
    )

    maximum_similarity = float(
        np.max(candidate_similarities)
    )

    total_observations = len(observations)

    candidate_consistency = (
        candidate_count / total_observations
    )

    return {
        "candidate": candidate,
        "candidate_count": candidate_count,
        "total_observations": total_observations,
        "average_similarity": average_similarity,
        "minimum_similarity": minimum_similarity,
        "maximum_similarity": maximum_similarity,
        "candidate_consistency": candidate_consistency
    }


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=" * 60)
print("Loading YOLOv8...")
print("=" * 60)

yolo_model = YOLO("yolov8n.pt")

print("YOLOv8 loaded successfully.")


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print()
print("=" * 60)
print("Loading InsightFace...")
print("=" * 60)

face_app = insightface.app.FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded successfully.")


# ============================================================
# LOAD FACE DATABASE
# ============================================================

print()
print("=" * 60)
print("Loading face database...")
print("=" * 60)

try:

    database = np.load(DATABASE_PATH)

    database_embeddings = database["embeddings"]
    database_person_ids = database["person_ids"]

except Exception as error:

    print("ERROR: Could not load face database.")
    print(error)
    exit()


print(
    "Database embeddings:",
    len(database_embeddings)
)

print(
    "Embedding shape:",
    database_embeddings.shape
)

print(
    "Person IDs:",
    database_person_ids
)


# ============================================================
# OPEN WEBCAM
# ============================================================

print()
print("=" * 60)
print("Opening webcam...")
print("=" * 60)

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print()
print("=" * 60)
print("Integrated tracking verification started.")
print("=" * 60)

print()
print("Initial match threshold:",
      INITIAL_MATCH_THRESHOLD)

print(
    "Required observations:",
    REQUIRED_OBSERVATIONS
)

print()
print("Press Q to quit.")
print()


# ============================================================
# FRAME COUNTER
# ============================================================

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


    if len(results) == 0:

        cv2.imshow(
            "Integrated Missing Person Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    result = results[0]


    # ========================================================
    # CHECK FOR DETECTIONS
    # ========================================================

    if result.boxes is None:

        cv2.imshow(
            "Integrated Missing Person Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    if result.boxes.id is None:

        cv2.imshow(
            "Integrated Missing Person Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    # ========================================================
    # GET YOLO DATA
    # ========================================================

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    track_ids = (
        result.boxes.id
        .cpu()
        .numpy()
        .astype(int)
    )

    classes = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )


    current_time = time.monotonic()

    current_track_ids = set()


    # ========================================================
    # PROCESS EACH DETECTED PERSON
    # ========================================================

    for box, track_id, class_id in zip(
        boxes,
        track_ids,
        classes
    ):

        # ----------------------------------------------------
        # Only process person class
        # ----------------------------------------------------

        if class_id != 0:
            continue


        current_track_ids.add(track_id)


        # ====================================================
        # CREATE / UPDATE TRACK
        # ====================================================

        if track_id not in tracks:

            tracks[track_id] = {

                "first_seen": current_time,

                "last_seen": current_time,

                "observations": [],

                "status": "COLLECTING"
            }

        else:

            tracks[track_id]["last_seen"] = current_time


        track = tracks[track_id]


        # ====================================================
        # PERSON BOUNDING BOX
        # ====================================================

        x1, y1, x2, y2 = box.astype(int)


        h, w = frame.shape[:2]


        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)


        if x2 <= x1 or y2 <= y1:
            continue


        # ====================================================
        # DRAW PERSON BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ====================================================
        # TRACK DURATION
        # ====================================================

        duration = (
            current_time
            - track["first_seen"]
        )


        # ====================================================
        # FACE PROCESSING
        # ====================================================

        if (
            frame_count % PROCESS_FACE_EVERY_N_FRAMES == 0
            and len(track["observations"])
            < REQUIRED_OBSERVATIONS
        ):


            # ------------------------------------------------
            # Crop person
            # ------------------------------------------------

            person_crop = frame[
                y1:y2,
                x1:x2
            ]


            if person_crop.size == 0:
                continue


            # ------------------------------------------------
            # Detect faces
            # ------------------------------------------------

            faces = face_app.get(
                person_crop
            )


            if len(faces) == 0:

                cv2.putText(
                    frame,
                    "No face",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                continue


            # ------------------------------------------------
            # Select largest face
            # ------------------------------------------------

            largest_face = max(
                faces,
                key=lambda face: (
                    face.bbox[2]
                    - face.bbox[0]
                )
                * (
                    face.bbox[3]
                    - face.bbox[1]
                )
            )


            # ------------------------------------------------
            # Face bounding box
            # ------------------------------------------------

            face_bbox = (
                largest_face.bbox
            )


            # ------------------------------------------------
            # Face quality check
            # ------------------------------------------------

            good_quality, blur_score = (
                check_face_quality(
                    person_crop,
                    face_bbox
                )
            )


            if not good_quality:

                cv2.putText(
                    frame,
                    "Poor face quality",
                    (x1, min(y2 + 20, h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

                continue


            # ------------------------------------------------
            # Get face embedding
            # ------------------------------------------------

            query_embedding = (
                largest_face.embedding
            )


            # =================================================
            # SEARCH DATABASE
            # =================================================

            similarities = []


            for database_embedding in (
                database_embeddings
            ):

                similarity = cosine_similarity(
                    query_embedding,
                    database_embedding
                )

                similarities.append(
                    similarity
                )


            if len(similarities) == 0:
                continue


            # ------------------------------------------------
            # Find best database match
            # ------------------------------------------------

            best_index = int(
                np.argmax(similarities)
            )


            best_similarity = float(
                similarities[best_index]
            )


            best_person_id = str(
                database_person_ids[
                    best_index
                ]
            )


            # =================================================
            # STORE OBSERVATION
            # =================================================

            track["observations"].append({

                "person_id":
                    best_person_id,

                "similarity":
                    best_similarity
            })


            # ------------------------------------------------
            # Keep only required observations
            # ------------------------------------------------

            if len(track["observations"]) > (
                REQUIRED_OBSERVATIONS
            ):

                track["observations"] = (
                    track["observations"]
                    [-REQUIRED_OBSERVATIONS:]
                )


            # =================================================
            # CALCULATE CURRENT STATISTICS
            # =================================================

            statistics = (
                calculate_track_statistics(
                    track["observations"]
                )
            )


            # =================================================
            # DISPLAY OBSERVATION
            # =================================================

            observation_number = len(
                track["observations"]
            )


            print(
                f"Track ID: {track_id} | "
                f"Observation: "
                f"{observation_number}/"
                f"{REQUIRED_OBSERVATIONS} | "
                f"Candidate: "
                f"{best_person_id} | "
                f"Similarity: "
                f"{best_similarity:.3f} | "
                f"Blur: "
                f"{blur_score:.0f}"
            )


            # =================================================
            # UPDATE TRACK STATUS
            # =================================================

            if (
                len(track["observations"])
                >= REQUIRED_OBSERVATIONS
            ):

                average_similarity = (
                    statistics[
                        "average_similarity"
                    ]
                )


                if (
                    average_similarity
                    >= INITIAL_MATCH_THRESHOLD
                ):

                    track["status"] = (
                        "POSSIBLE MATCH - "
                        "HUMAN VERIFICATION REQUIRED"
                    )

                else:

                    track["status"] = (
                        "NO MATCH"
                    )

            else:

                track["status"] = (
                    "COLLECTING"
                )


    # ========================================================
    # DISPLAY INFORMATION FOR EACH ACTIVE TRACK
    # ========================================================

    for track_id in current_track_ids:

        if track_id not in tracks:
            continue


        track = tracks[track_id]


        # ----------------------------------------------------
        # Track duration
        # ----------------------------------------------------

        duration = (
            current_time
            - track["first_seen"]
        )


        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        statistics = (
            calculate_track_statistics(
                track["observations"]
            )
        )


        # ----------------------------------------------------
        # Find corresponding person box
        # ----------------------------------------------------

        matching_box = None

        for box, detected_track_id, class_id in zip(
            boxes,
            track_ids,
            classes
        ):

            if detected_track_id == track_id:

                matching_box = box

                break


        if matching_box is None:
            continue


        bx1, by1, bx2, by2 = (
            matching_box.astype(int)
        )


        # ----------------------------------------------------
        # Basic display text
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Track ID: {track_id}",
            (bx1, max(by1 - 50, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Duration: {duration:.1f}s",
            (bx1, max(by1 - 30, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY STATISTICS
        # ====================================================

        if statistics is not None:

            candidate = statistics[
                "candidate"
            ]

            average_similarity = statistics[
                "average_similarity"
            ]

            minimum_similarity = statistics[
                "minimum_similarity"
            ]

            maximum_similarity = statistics[
                "maximum_similarity"
            ]

            consistency = statistics[
                "candidate_consistency"
            ]


            # ------------------------------------------------
            # Candidate
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Candidate: {candidate}",
                (bx1, min(by2 + 20, h - 80)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # Average similarity
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Avg: {average_similarity:.3f}",
                (bx1, min(by2 + 40, h - 60)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # Observation count
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Obs: "
                f"{len(track['observations'])}/"
                f"{REQUIRED_OBSERVATIONS}",
                (bx1, min(by2 + 60, h - 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # Consistency
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Consistency: "
                f"{consistency * 100:.0f}%",
                (bx1, min(by2 + 80, h - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )


        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        status = track["status"]


        if status == "NO MATCH":

            status_text = "NO MATCH"


        elif status.startswith(
            "POSSIBLE MATCH"
        ):

            status_text = (
                "POSSIBLE MATCH"
            )


        else:

            status_text = (
                "COLLECTING"
            )


        cv2.putText(
            frame,
            status_text,
            (bx1, max(by1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


    # ========================================================
    # CHECK FOR EXPIRED TRACKS
    # ========================================================

    expired_tracks = []


    for track_id, track in tracks.items():

        if track_id in current_track_ids:
            continue


        time_since_last_seen = (
            current_time
            - track["last_seen"]
        )


        if time_since_last_seen > TRACK_TIMEOUT:

            expired_tracks.append(
                track_id
            )


    # ========================================================
    # PROCESS FINISHED TRACKS
    # ========================================================

    for track_id in expired_tracks:

        track = tracks[track_id]


        print()
        print("=" * 60)
        print(
            f"Track {track_id} finished."
        )
        print("=" * 60)


        observations = (
            track["observations"]
        )


        # ----------------------------------------------------
        # No observations
        # ----------------------------------------------------

        if len(observations) == 0:

            print(
                "No usable face observations."
            )

            print(
                "Status: "
                "INSUFFICIENT OBSERVATIONS"
            )

            print()

            del tracks[track_id]

            continue


        # ----------------------------------------------------
        # Calculate final statistics
        # ----------------------------------------------------

        statistics = (
            calculate_track_statistics(
                observations
            )
        )


        candidate = statistics[
            "candidate"
        ]

        candidate_count = statistics[
            "candidate_count"
        ]

        total_observations = statistics[
            "total_observations"
        ]

        average_similarity = statistics[
            "average_similarity"
        ]

        minimum_similarity = statistics[
            "minimum_similarity"
        ]

        maximum_similarity = statistics[
            "maximum_similarity"
        ]

        consistency = statistics[
            "candidate_consistency"
        ]


        # ====================================================
        # PRINT FINAL RESULTS
        # ====================================================

        print(
            "Candidate:",
            candidate
        )


        print(
            "Observations:",
            total_observations
        )


        print(
            "Candidate observations:",
            candidate_count
        )


        print(
            "Candidate consistency:",
            f"{consistency * 100:.1f}%"
        )


        print(
            "Average similarity:",
            f"{average_similarity:.3f}"
        )


        print(
            "Minimum similarity:",
            f"{minimum_similarity:.3f}"
        )


        print(
            "Maximum similarity:",
            f"{maximum_similarity:.3f}"
        )


        print(
            "Initial threshold:",
            f"{INITIAL_MATCH_THRESHOLD:.3f}"
        )


        # ====================================================
        # FINAL DECISION
        # ====================================================

        if (
            total_observations
            < REQUIRED_OBSERVATIONS
        ):

            final_status = (
                "INSUFFICIENT OBSERVATIONS"
            )


        elif (
            average_similarity
            >= INITIAL_MATCH_THRESHOLD
        ):

            final_status = (
                "POSSIBLE MATCH - "
                "HUMAN VERIFICATION REQUIRED"
            )


        else:

            final_status = (
                "NO MATCH"
            )


        print(
            "Status:",
            final_status
        )


        print("=" * 60)
        print()


        # ----------------------------------------------------
        # Remove completed track
        # ----------------------------------------------------

        del tracks[track_id]


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Integrated Missing Person Detection",
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


print()
print("=" * 60)
print("Integrated tracking verification stopped.")
print("=" * 60)