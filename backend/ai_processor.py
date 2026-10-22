import os
import time
from collections import defaultdict, Counter

import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from face_quality import check_face_quality
from embedding_utils import cosine_similarity
from backend.database import get_connection


# ============================================================
# SETTINGS
# ============================================================

PROCESS_FACE_EVERY_N_FRAMES = 5
REQUIRED_OBSERVATIONS = 10
TRACK_TIMEOUT = 2.0

# Provisional calibration threshold.
# This is NOT the final project threshold.
INITIAL_MATCH_THRESHOLD = 0.60


# ============================================================
# PATHS
# ============================================================

DATABASE_PATH = "data/face_database.npz"

# Folder where possible-match camera images will be saved
ALERTS_FOLDER = "data/alerts"

os.makedirs(
    ALERTS_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLOv8 model...")

yolo_model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded successfully.")


print("Loading InsightFace model...")

face_app = FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace model loaded successfully.")


# ============================================================
# FACE DATABASE
# ============================================================

def load_face_database():

    global database_embeddings
    global database_ids

    if not os.path.exists(DATABASE_PATH):

        raise FileNotFoundError(
            f"Face database not found: {DATABASE_PATH}"
        )

    database = np.load(
        DATABASE_PATH,
        allow_pickle=True
    )

    database_embeddings = database["embeddings"]

    database_ids = database["person_ids"]

    print(
        f"Face database loaded: "
        f"{len(database_embeddings)} embeddings"
    )


def reload_face_database():

    """
    Reload the face database from disk.

    This is called after a new missing person
    is added to the authorized database.
    """

    load_face_database()

    print(
        "Face database reloaded successfully."
    )


# Load database when the AI processor starts
load_face_database()


# ============================================================
# TRACK STORAGE
# ============================================================

tracks = defaultdict(
    lambda: {

        "first_seen":
            time.monotonic(),

        "last_seen":
            time.monotonic(),

        "observations":
            [],

        "status":
            "COLLECTING",

        # ----------------------------------------------------
        # BEST CAMERA FRAME
        # ----------------------------------------------------

        "best_frame":
            None,

        "best_blur_score":
            -1.0,

        "best_similarity":
            0.0,

        "best_person_id":
            None,

        "alert_image_path":
            None,

        "alert_id":
            None
    }
)


# ============================================================
# DATABASE SEARCH
# ============================================================

def search_database(face_embedding):

    best_person_id = None

    best_similarity = -1.0

    for stored_embedding, person_id in zip(
        database_embeddings,
        database_ids
    ):

        similarity = cosine_similarity(
            face_embedding,
            stored_embedding
        )

        if similarity > best_similarity:

            best_similarity = similarity

            best_person_id = str(
                person_id
            )

    return (
        best_person_id,
        best_similarity
    )


# ============================================================
# SAVE BEST CAMERA FRAME
# ============================================================

def save_best_camera_frame(
    track_id,
    frame,
    blur_score,
    person_id,
    similarity
):

    """

    Save the best-quality camera frame for a track.

    The frame is only saved when its blur score is
    better than the previously stored frame.

    Returns the saved image path.
    """

    track_folder = os.path.join(
        ALERTS_FOLDER,
        f"track_{track_id}"
    )

    os.makedirs(
        track_folder,
        exist_ok=True
    )

    image_path = os.path.join(
        track_folder,
        "captured.jpg"
    )

    success = cv2.imwrite(
        image_path,
        frame
    )

    if success:

        return image_path

    return None


# ============================================================
# SAVE ALERT TO DATABASE
# ============================================================

def save_alert_to_database(
    track_id,
    person_id,
    similarity,
    observations,
    camera_id,
    camera_location,
    captured_image_path
):
    """
    Save a possible-match alert permanently in SQLite.
    """

    try:
        connection = get_connection()

        cursor = connection.execute(
            """
            INSERT INTO alerts (
                track_id,
                person_id,
                similarity,
                observations,
                camera_id,
                camera_location,
                captured_image_path,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track_id,
                person_id,
                float(similarity),
                int(observations),
                camera_id,
                camera_location,
                captured_image_path,
                "PENDING"
            )
        )

        connection.commit()

        alert_id = cursor.lastrowid

        connection.close()

        print("💾 ALERT SAVED TO DATABASE")
        print(f"Alert ID: {alert_id}")
        print(f"Track ID: {track_id}")
        print(f"Candidate: {person_id}")
        print(f"Similarity: {similarity:.3f}")
        print(f"Status: PENDING")
        print()

        return alert_id

    except Exception as e:
        print(f"❌ Failed to save alert to database: {e}")
        return None

    
# ============================================================
# TRACK STATISTICS
# ============================================================

def calculate_track_statistics(track):

    observations = track["observations"]

    if not observations:

        return {

            "candidate":
                None,

            "candidate_count":
                0,

            "total_observations":
                0,

            "average_similarity":
                0.0,

            "minimum_similarity":
                0.0,

            "maximum_similarity":
                0.0,

            "consistency":
                0.0
        }

    candidates = [

        observation["person_id"]

        for observation in observations

    ]

    counter = Counter(
        candidates
    )

    candidate, candidate_count = (
        counter.most_common(1)[0]
    )

    candidate_similarities = [

        observation["similarity"]

        for observation in observations

        if observation["person_id"] == candidate

    ]

    average_similarity = float(
        np.mean(
            candidate_similarities
        )
    )

    minimum_similarity = float(
        np.min(
            candidate_similarities
        )
    )

    maximum_similarity = float(
        np.max(
            candidate_similarities
        )
    )

    consistency = (

        candidate_count /

        len(observations)

    ) * 100

    return {

        "candidate":
            candidate,

        "candidate_count":
            candidate_count,

        "total_observations":
            len(observations),

        "average_similarity":
            average_similarity,

        "minimum_similarity":
            minimum_similarity,

        "maximum_similarity":
            maximum_similarity,

        "consistency":
            consistency
    }


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    frame,
    frame_count
):

    current_time = time.monotonic()

    results = yolo_model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=0.5,

        classes=[0],

        verbose=False
    )

    result = results[0]

    active_track_ids = set()

    if result.boxes is None:

        return frame

    boxes = result.boxes

    if boxes.id is None:

        return frame

    xyxy = boxes.xyxy.cpu().numpy()

    track_ids = (

        boxes.id
        .cpu()
        .numpy()
        .astype(int)

    )


    # ========================================================
    # PROCESS EACH PERSON
    # ========================================================

    for box, track_id in zip(
        xyxy,
        track_ids
    ):

        track_id = int(
            track_id
        )

        active_track_ids.add(
            track_id
        )

        x1, y1, x2, y2 = (
            box.astype(int)
        )


        # ----------------------------------------------------
        # MAKE SURE COORDINATES STAY INSIDE FRAME
        # ----------------------------------------------------

        h, w = frame.shape[:2]

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            w,
            x2
        )

        y2 = min(
            h,
            y2
        )


        # ----------------------------------------------------
        # CREATE TRACK IF NEW
        # ----------------------------------------------------

        if track_id not in tracks:

            tracks[track_id] = {

                "first_seen":
                    current_time,

                "last_seen":
                    current_time,

                "observations":
                    [],

                "status":
                    "COLLECTING",

                "best_frame":
                    None,

                "best_blur_score":
                    -1.0,

                "best_similarity":
                    0.0,

                "best_person_id":
                    None,

                "alert_image_path":
                    None,

                "alert_id":
                    None
            }


        tracks[track_id][
            "last_seen"
        ] = current_time

        track = tracks[
            track_id
        ]


        # ====================================================
        # PERSON BOX
        # ====================================================

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2
        )


        # ====================================================
        # FACE PROCESSING
        # ====================================================

        if (

            frame_count %
            PROCESS_FACE_EVERY_N_FRAMES == 0

            and

            len(
                track["observations"]
            )
            <
            REQUIRED_OBSERVATIONS

        ):

            person_crop = frame[
                y1:y2,
                x1:x2
            ]

            if person_crop.size > 0:

                faces = face_app.get(
                    person_crop
                )

                if len(faces) > 0:

                    # ----------------------------------------
                    # SELECT LARGEST FACE
                    # ----------------------------------------

                    face = max(

                        faces,

                        key=lambda f:

                        (

                            f.bbox[2] -
                            f.bbox[0]

                        )
                        *
                        (

                            f.bbox[3] -
                            f.bbox[1]

                        )
                    )

                    bbox = face.bbox


                    # ----------------------------------------
                    # FACE QUALITY CHECK
                    # ----------------------------------------

                    good_quality, blur_score = (
                        check_face_quality(

                            person_crop,

                            bbox
                        )
                    )


                    if good_quality:

                        # ------------------------------------
                        # FACE EMBEDDING
                        # ------------------------------------

                        embedding = (
                            face.embedding
                        )


                        # ------------------------------------
                        # SEARCH DATABASE
                        # ------------------------------------

                        person_id, similarity = (
                            search_database(
                                embedding
                            )
                        )


                        if person_id is not None:

                            # --------------------------------
                            # STORE OBSERVATION
                            # --------------------------------

                            track[
                                "observations"
                            ].append(

                                {

                                    "person_id":
                                        person_id,

                                    "similarity":
                                        similarity,

                                    "blur_score":
                                        blur_score
                                }
                            )


                            # --------------------------------
                            # KEEP BEST CAMERA FRAME
                            # --------------------------------

                            if (

                                blur_score
                                >
                                track[
                                    "best_blur_score"
                                ]

                            ):

                                track[
                                    "best_blur_score"
                                ] = (
                                    blur_score
                                )

                                track[
                                    "best_similarity"
                                ] = (
                                    similarity
                                )

                                track[
                                    "best_person_id"
                                ] = (
                                    person_id
                                )

                                # --------------------------------
                                # COPY CURRENT FRAME
                                # --------------------------------

                                track[
                                    "best_frame"
                                ] = frame.copy()


                            print(

                                f"Track ID: "
                                f"{track_id} | "

                                f"Candidate: "
                                f"{person_id} | "

                                f"Similarity: "
                                f"{similarity:.3f} | "

                                f"Blur: "
                                f"{blur_score:.0f} | "

                                f"Observation: "

                                f"{len(track['observations'])}/"

                                f"{REQUIRED_OBSERVATIONS}"
                            )


        # ====================================================
        # TRACK STATUS
        # ====================================================

        if (

            len(
                track["observations"]
            )
            <
            REQUIRED_OBSERVATIONS

        ):

            track[
                "status"
            ] = (
                "COLLECTING"
            )

        else:

            statistics = (
                calculate_track_statistics(
                    track
                )
            )

            if (

                statistics[
                    "average_similarity"
                ]

                >=

                INITIAL_MATCH_THRESHOLD

            ):

                track[
                    "status"
                ] = (

                    "POSSIBLE MATCH - "

                    "HUMAN VERIFICATION REQUIRED"

                )

            else:

                track[
                    "status"
                ] = (
                    "NO MATCH"
                )


        # ====================================================
        # SAVE BEST FRAME WHEN POSSIBLE MATCH IS REACHED
        # ====================================================

        if (

            track["status"]

            ==

            "POSSIBLE MATCH - "
            "HUMAN VERIFICATION REQUIRED"

            and

            track["alert_image_path"]
            is None

            and

            track["best_frame"] is not None

        ):

            saved_path = (
                save_best_camera_frame(

                    track_id,

                    track[
                        "best_frame"
                    ],

                    track[
                        "best_blur_score"
                    ],

                    track[
                        "best_person_id"
                    ],

                    track[
                        "best_similarity"
                    ]
                )
            )


            if saved_path is not None:

                track[
                    "alert_image_path"
                ] = saved_path

                # ------------------------------------------------
                # SAVE PERSISTENT ALERT
                # ------------------------------------------------

                if track["alert_id"] is None:

                    alert_id = save_alert_to_database(
                        track_id=track_id,
                        person_id=track["best_person_id"],
                        similarity=track["best_similarity"],
                        observations=len(track["observations"]),
                        camera_id="camera_01",
                        camera_location="Main Entrance",
                        captured_image_path=saved_path
                    )

                    track["alert_id"] = alert_id

                print()
                print(
                    "🚨 POSSIBLE MATCH CAMERA IMAGE SAVED"
                )

                print(
                    f"Track ID: "
                    f"{track_id}"
                )

                print(
                    f"Candidate: "
                    f"{track['best_person_id']}"
                )

                print(
                    f"Similarity: "
                    f"{track['best_similarity']:.3f}"
                )

                print(
                    f"Blur score: "
                    f"{track['best_blur_score']:.0f}"
                )

                print(
                    f"Image: "
                    f"{saved_path}"
                )

                print()


        # ====================================================
        # DISPLAY INFORMATION
        # ====================================================

        statistics = (
            calculate_track_statistics(
                track
            )
        )

        duration = (

            current_time -

            track[
                "first_seen"
            ]

        )

        text_y = max(

            y1 - 10,

            20
        )


        # ----------------------------------------------------
        # TRACK ID
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Track ID: {track_id}",

            (x1, text_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # CANDIDATE
        # ----------------------------------------------------

        if (

            statistics[
                "candidate"
            ]

            is not None

        ):

            cv2.putText(

                frame,

                (

                    f"Candidate: "

                    f"{statistics['candidate']}"

                ),

                (x1, text_y + 22),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (0, 255, 255),

                2
            )


            # ------------------------------------------------
            # SIMILARITY
            # ------------------------------------------------

            cv2.putText(

                frame,

                (

                    f"Similarity: "

                    f"{statistics['average_similarity']:.3f}"

                ),

                (x1, text_y + 44),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (0, 255, 255),

                2
            )


            # ------------------------------------------------
            # OBSERVATIONS
            # ------------------------------------------------

            cv2.putText(

                frame,

                (

                    f"Observations: "

                    f"{statistics['total_observations']}/"

                    f"{REQUIRED_OBSERVATIONS}"

                ),

                (x1, text_y + 66),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (255, 255, 0),

                2
            )


            # ------------------------------------------------
            # CONSISTENCY
            # ------------------------------------------------

            cv2.putText(

                frame,

                (

                    f"Consistency: "

                    f"{statistics['consistency']:.0f}%"

                ),

                (x1, text_y + 88),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (255, 255, 0),

                2
            )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        cv2.putText(

            frame,

            track[
                "status"
            ],

            (x1, text_y + 110),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0, 255, 0),

            2
        )


        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Duration: {duration:.1f}s",

            (x1, text_y + 132),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (255, 255, 255),

            2
        )


    # ========================================================
    # REMOVE EXPIRED TRACKS
    # ========================================================

    expired_tracks = []

    for track_id, track in list(
        tracks.items()
    ):

        if track_id in active_track_ids:

            continue

        time_since_seen = (

            current_time -

            track[
                "last_seen"
            ]

        )

        if (

            time_since_seen

            >

            TRACK_TIMEOUT

        ):

            expired_tracks.append(
                track_id
            )


    # ========================================================
    # PRINT FINAL TRACK RESULT
    # ========================================================

    for track_id in expired_tracks:

        track = tracks[
            track_id
        ]

        statistics = (
            calculate_track_statistics(
                track
            )
        )

        print()

        print(
            f"Track {track_id} finished."
        )

        print(
            f"Candidate: "
            f"{statistics['candidate']}"
        )

        print(
            f"Observations: "
            f"{statistics['total_observations']}"
        )

        print(
            f"Candidate observations: "
            f"{statistics['candidate_count']}"
        )

        print(
            f"Candidate consistency: "
            f"{statistics['consistency']:.1f}%"
        )

        print(
            f"Average similarity: "
            f"{statistics['average_similarity']:.3f}"
        )

        print(
            f"Minimum similarity: "
            f"{statistics['minimum_similarity']:.3f}"
        )

        print(
            f"Maximum similarity: "
            f"{statistics['maximum_similarity']:.3f}"
        )

        print(
            f"Initial threshold: "
            f"{INITIAL_MATCH_THRESHOLD:.3f}"
        )

        print(
            f"Status: "
            f"{track['status']}"
        )

        if track[
            "alert_image_path"
        ] is not None:

            print(
                f"Alert image: "
                f"{track['alert_image_path']}"
            )

        print()

        del tracks[
            track_id
        ]


    return frame


# ============================================================
# GET CURRENT DETECTION RESULTS
# ============================================================

def get_detection_results():

    results = []

    for track_id, track in tracks.items():

        statistics = (
            calculate_track_statistics(
                track
            )
        )

        results.append(

            {

                "track_id":
                    track_id,

                "candidate":
                    statistics[
                        "candidate"
                    ],

                "similarity":
                    round(

                        statistics[
                            "average_similarity"
                        ],

                        3
                    ),

                "observations":
                    statistics[
                        "total_observations"
                    ],

                "consistency":
                    round(

                        statistics[
                            "consistency"
                        ],

                        1
                    ),

                "status":
                    track[
                        "status"
                    ],

                "alert_image_path":
                    track[
                        "alert_image_path"
                    ]
            }
        )

    return results