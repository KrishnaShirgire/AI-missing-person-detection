from backend.database import get_connection
from backend.face_database_manager import add_person_to_database

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import cv2
import os
import shutil

from backend.ai_processor import (
    process_frame,
    get_detection_results,
    reload_face_database
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Missing Person Detection API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SERVE DATA / IMAGES
# ============================================================

# Allows frontend to access:
# /data/alerts/...
# /data/persons/...

if os.path.exists("data"):
    app.mount(
        "/data",
        StaticFiles(directory="data"),
        name="data"
    )


# ============================================================
# HUMAN VERIFICATION STORE
# ============================================================

# Temporary in-memory verification store.
# Kept for compatibility with the existing detection API.

verification_results = {}


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Missing Person Detection Backend is running"
    }


# ============================================================
# MISSING PERSONS - GET ALL
# ============================================================

@app.get("/api/v1/missing-persons")
def get_missing_persons():

    connection = get_connection()

    try:

        persons = connection.execute(
            """
            SELECT *
            FROM missing_persons
            ORDER BY id
            """
        ).fetchall()

        result = []

        for person in persons:

            person_data = dict(person)

            # -----------------------------------------------
            # CONVERT PERSON IMAGE PATH TO FRONTEND URL
            # -----------------------------------------------

            image_path = person_data.get("image_path")

            if image_path:

                normalized_path = (
                    image_path
                    .replace("\\", "/")
                )

                # If image_path points to the directory,
                # find the actual image inside it.
                if normalized_path.startswith("data/"):
                    relative_path = normalized_path[5:]
                else:
                    relative_path = normalized_path

                # Check whether image_path is already a file
                full_path = image_path

                if os.path.isfile(full_path):

                    person_data["person_image_url"] = (
                        "/data/" + relative_path
                    )

                else:

                    # image_path may be a directory.
                    # Find first supported image.
                    directory = image_path

                    found_image = None

                    if os.path.isdir(directory):

                        for filename in os.listdir(directory):

                            if filename.lower().endswith(
                                (".jpg", ".jpeg", ".png")
                            ):

                                found_image = filename
                                break

                    if found_image:

                        directory_normalized = (
                            directory
                            .replace("\\", "/")
                        )

                        if directory_normalized.startswith("data/"):
                            directory_normalized = (
                                directory_normalized[5:]
                            )

                        person_data["person_image_url"] = (
                            "/data/"
                            + directory_normalized
                            + "/"
                            + found_image
                        )

                    else:

                        person_data["person_image_url"] = None

            else:

                person_data["person_image_url"] = None

            result.append(person_data)

        return {
            "persons": result
        }

    finally:

        connection.close()


# ============================================================
# MISSING PERSONS - REGISTER
# ============================================================

@app.post("/api/v1/missing-persons")
async def register_missing_person(

    id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    last_seen_location: str = Form(...),
    last_seen_date: str = Form(...),
    description: str = Form(""),
    photo: UploadFile = File(...)

):

    connection = get_connection()

    try:

        # ----------------------------------------------------
        # CHECK IF PERSON ID ALREADY EXISTS
        # ----------------------------------------------------

        existing_person = connection.execute(
            "SELECT id FROM missing_persons WHERE id = ?",
            (id,)
        ).fetchone()

        if existing_person:

            raise HTTPException(
                status_code=400,
                detail=f"Person ID {id} already exists"
            )


        # ----------------------------------------------------
        # CREATE PERSON DIRECTORY
        # ----------------------------------------------------

        person_directory = os.path.join(
            "data",
            "persons",
            id
        )

        os.makedirs(
            person_directory,
            exist_ok=True
        )


        # ----------------------------------------------------
        # CHECK FILE NAME
        # ----------------------------------------------------

        original_filename = photo.filename

        if not original_filename:

            raise HTTPException(
                status_code=400,
                detail="Photo filename is missing"
            )


        # ----------------------------------------------------
        # GET FILE EXTENSION
        # ----------------------------------------------------

        file_extension = os.path.splitext(
            original_filename
        )[1].lower()

        if file_extension not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG and PNG images are allowed"
            )


        # ----------------------------------------------------
        # SAVE PHOTO
        # ----------------------------------------------------

        image_path = os.path.join(
            person_directory,
            f"image1{file_extension}"
        )

        with open(
            image_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                photo.file,
                buffer
            )


        # ----------------------------------------------------
        # GENERATE FACE EMBEDDING
        # ----------------------------------------------------

        try:

            embedding_result = add_person_to_database(
                person_id=id,
                image_path=image_path
            )

            reload_face_database()

        except Exception as embedding_error:

            # Remove image if face embedding failed.

            if os.path.exists(image_path):

                os.remove(image_path)

            try:

                os.rmdir(person_directory)

            except OSError:

                pass

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Face embedding failed: "
                    f"{str(embedding_error)}"
                )
            )


        # ----------------------------------------------------
        # SAVE PERSON DETAILS TO DATABASE
        # ----------------------------------------------------

        connection.execute(
            """
            INSERT INTO missing_persons
            (
                id,
                name,
                age,
                gender,
                last_seen_location,
                last_seen_date,
                description,
                image_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                name,
                age,
                gender,
                last_seen_location,
                last_seen_date,
                description,
                person_directory
            )
        )

        connection.commit()


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "message": (
                "Missing person registered successfully"
            ),

            "person": {

                "id": id,

                "name": name,

                "age": age,

                "gender": gender,

                "last_seen_location": (
                    last_seen_location
                ),

                "last_seen_date": (
                    last_seen_date
                ),

                "description": description,

                "image_path": image_path

            },

            "face_embedding": {

                "created": True,

                "embedding_size": (
                    embedding_result["embedding_size"]
                ),

                "total_embeddings": (
                    embedding_result["total_embeddings"]
                )

            }

        }

    finally:

        connection.close()


# ============================================================
# CAMERA API
# ============================================================

@app.get("/api/v1/cameras")
def get_cameras():

    return [

        {
            "id": "camera_01",
            "name": "Webcam",
            "location": "Test Lab",
            "rtsp_url": "",
            "status": "online",
            "filters": [],
            "model": "YOLOv8"
        }

    ]


# ============================================================
# STREAM STATUS
# ============================================================

@app.get("/api/v1/streaming/")
def get_streaming_status():

    return {

        "active_streams": [

            {
                "camera_id": "camera_01",
                "status": "online"
            }

        ]

    }


# ============================================================
# STREAM URLS
# ============================================================

@app.get("/api/v1/streaming/stream-urls")
def get_stream_urls():

    return {

        "streams": [

            {
                "camera_id": "camera_01",

                "stream_url": (
                    "http://127.0.0.1:8000/"
                    "api/v1/streaming/camera_01"
                ),

                "status": "online"
            }

        ]

    }


# ============================================================
# DETECTION RESULTS
# ============================================================

@app.get("/api/v1/detections")
def get_detections():

    detections = get_detection_results()


    # --------------------------------------------------------
    # ADD HUMAN VERIFICATION RESULT
    # --------------------------------------------------------

    for detection in detections:

        track_id = detection["track_id"]

        if track_id in verification_results:

            detection["verification"] = (
                verification_results[track_id]
            )


    return {

        "detections": detections

    }


# ============================================================
# CONFIRM DETECTION
# ============================================================

@app.post("/api/v1/detections/{track_id}/confirm")
def confirm_detection(track_id: int):

    detections = get_detection_results()

    detection = next(
        (
            item
            for item in detections
            if item["track_id"] == track_id
        ),
        None
    )

    if detection is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Track ID {track_id} not found"
            )
        )


    verification_results[track_id] = {

        "status": "CONFIRMED BY HUMAN",

        "candidate": detection["candidate"]

    }


    return {

        "message": (
            "Detection confirmed successfully"
        ),

        "track_id": track_id,

        "candidate": detection["candidate"],

        "verification": (
            verification_results[track_id]
        )

    }


# ============================================================
# REJECT DETECTION
# ============================================================

@app.post("/api/v1/detections/{track_id}/reject")
def reject_detection(track_id: int):

    detections = get_detection_results()

    detection = next(
        (
            item
            for item in detections
            if item["track_id"] == track_id
        ),
        None
    )

    if detection is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Track ID {track_id} not found"
            )
        )


    verification_results[track_id] = {

        "status": "REJECTED BY HUMAN",

        "candidate": detection["candidate"]

    }


    return {

        "message": (
            "Detection rejected successfully"
        ),

        "track_id": track_id,

        "candidate": detection["candidate"],

        "verification": (
            verification_results[track_id]
        )

    }


# ============================================================
# HELPER - CREATE PERSON IMAGE URL
# ============================================================

def create_person_image_url(image_path):

    if not image_path:
        return None

    normalized_path = (
        image_path
        .replace("\\", "/")
    )

    # If the stored path starts with data/,
    # remove it because /data is added by StaticFiles.
    if normalized_path.startswith("data/"):

        normalized_path = normalized_path[5:]


    # --------------------------------------------------------
    # CASE 1: image_path itself is an image file
    # --------------------------------------------------------

    if os.path.isfile(image_path):

        return "/data/" + normalized_path


    # --------------------------------------------------------
    # CASE 2: image_path is a directory
    # --------------------------------------------------------

    if os.path.isdir(image_path):

        for filename in os.listdir(image_path):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                directory_normalized = (
                    image_path
                    .replace("\\", "/")
                )

                if directory_normalized.startswith("data/"):

                    directory_normalized = (
                        directory_normalized[5:]
                    )

                return (
                    "/data/"
                    + directory_normalized
                    + "/"
                    + filename
                )


    return None


# ============================================================
# HELPER - CREATE CAPTURED IMAGE URL
# ============================================================

def create_captured_image_url(image_path):

    if not image_path:
        return None

    normalized_path = (
        image_path
        .replace("\\", "/")
    )

    if normalized_path.startswith("data/"):

        normalized_path = normalized_path[5:]

    return "/data/" + normalized_path


# ============================================================
# ALERTS - GET ALL
# ============================================================

@app.get("/api/v1/alerts")
def get_alerts():

    connection = get_connection()

    try:

        # ----------------------------------------------------
        # GET ALERT + MISSING PERSON INFORMATION
        # ----------------------------------------------------

        rows = connection.execute(
            """
            SELECT

                alerts.id,
                alerts.track_id,
                alerts.person_id,
                alerts.similarity,
                alerts.observations,
                alerts.camera_id,
                alerts.camera_location,
                alerts.captured_image_path,
                alerts.status,
                alerts.created_at,
                alerts.verified_at,

                missing_persons.name,
                missing_persons.age,
                missing_persons.gender,
                missing_persons.last_seen_location,
                missing_persons.last_seen_date,
                missing_persons.description,
                missing_persons.image_path

            FROM alerts

            LEFT JOIN missing_persons

            ON alerts.person_id = missing_persons.id

            ORDER BY alerts.id DESC
            """
        ).fetchall()


        result = []


        for row in rows:

            alert_data = dict(row)


            # ------------------------------------------------
            # CAPTURED CAMERA IMAGE
            # ------------------------------------------------

            alert_data["captured_image_url"] = (
                create_captured_image_url(
                    alert_data.get(
                        "captured_image_path"
                    )
                )
            )


            # ------------------------------------------------
            # DATABASE PERSON IMAGE
            # ------------------------------------------------

            alert_data["person_image_url"] = (
                create_person_image_url(
                    alert_data.get(
                        "image_path"
                    )
                )
            )


            result.append(alert_data)


        return {

            "alerts": result

        }

    finally:

        connection.close()


# ============================================================
# ALERT - GET SINGLE ALERT
# ============================================================

@app.get("/api/v1/alerts/{alert_id}")
def get_alert(alert_id: int):

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT

                alerts.id,
                alerts.track_id,
                alerts.person_id,
                alerts.similarity,
                alerts.observations,
                alerts.camera_id,
                alerts.camera_location,
                alerts.captured_image_path,
                alerts.status,
                alerts.created_at,
                alerts.verified_at,

                missing_persons.name,
                missing_persons.age,
                missing_persons.gender,
                missing_persons.last_seen_location,
                missing_persons.last_seen_date,
                missing_persons.description,
                missing_persons.image_path

            FROM alerts

            LEFT JOIN missing_persons

            ON alerts.person_id = missing_persons.id

            WHERE alerts.id = ?

            """,
            (alert_id,)
        ).fetchone()


        if row is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Alert ID {alert_id} not found"
                )
            )


        alert_data = dict(row)


        # ----------------------------------------------------
        # IMAGE URLS
        # ----------------------------------------------------

        alert_data["captured_image_url"] = (
            create_captured_image_url(
                alert_data.get(
                    "captured_image_path"
                )
            )
        )


        alert_data["person_image_url"] = (
            create_person_image_url(
                alert_data.get(
                    "image_path"
                )
            )
        )


        return {

            "alert": alert_data

        }

    finally:

        connection.close()


# ============================================================
# ALERT - CONFIRM
# ============================================================

@app.post("/api/v1/alerts/{alert_id}/confirm")
def confirm_alert(alert_id: int):

    connection = get_connection()

    try:

        # ----------------------------------------------------
        # CHECK ALERT
        # ----------------------------------------------------

        alert = connection.execute(
            """
            SELECT *
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,)
        ).fetchone()


        if alert is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Alert ID {alert_id} not found"
                )
            )


        # ----------------------------------------------------
        # UPDATE ALERT
        # ----------------------------------------------------

        connection.execute(
            """
            UPDATE alerts

            SET
                status = ?,
                verified_at = CURRENT_TIMESTAMP

            WHERE id = ?
            """,
            (
                "CONFIRMED",
                alert_id
            )
        )

        connection.commit()


        # ----------------------------------------------------
        # RETURN UPDATED ALERT
        # ----------------------------------------------------

        updated_alert = connection.execute(
            """
            SELECT *
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,)
        ).fetchone()


        return {

            "message": (
                "Alert confirmed successfully"
            ),

            "alert": dict(updated_alert)

        }

    finally:

        connection.close()


# ============================================================
# ALERT - REJECT
# ============================================================

@app.post("/api/v1/alerts/{alert_id}/reject")
def reject_alert(alert_id: int):

    connection = get_connection()

    try:

        # ----------------------------------------------------
        # CHECK ALERT
        # ----------------------------------------------------

        alert = connection.execute(
            """
            SELECT *
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,)
        ).fetchone()


        if alert is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Alert ID {alert_id} not found"
                )
            )


        # ----------------------------------------------------
        # UPDATE ALERT
        # ----------------------------------------------------

        connection.execute(
            """
            UPDATE alerts

            SET
                status = ?,
                verified_at = CURRENT_TIMESTAMP

            WHERE id = ?
            """,
            (
                "REJECTED",
                alert_id
            )
        )

        connection.commit()


        # ----------------------------------------------------
        # RETURN UPDATED ALERT
        # ----------------------------------------------------

        updated_alert = connection.execute(
            """
            SELECT *
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,)
        ).fetchone()


        return {

            "message": (
                "Alert rejected successfully"
            ),

            "alert": dict(updated_alert)

        }

    finally:

        connection.close()


# ============================================================
# VIDEO FRAME GENERATOR
# ============================================================

def generate_frames():

    camera = cv2.VideoCapture(0)


    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return


    frame_count = 0


    try:

        while True:

            success, frame = camera.read()


            if not success:

                break


            frame_count += 1


            # ------------------------------------------------
            # SEND FRAME TO AI PROCESSOR
            # ------------------------------------------------

            frame = process_frame(
                frame,
                frame_count
            )


            # ------------------------------------------------
            # CONVERT FRAME TO JPEG
            # ------------------------------------------------

            success, buffer = cv2.imencode(
                ".jpg",
                frame
            )


            if not success:

                continue


            frame_bytes = buffer.tobytes()


            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )


    finally:

        camera.release()


# ============================================================
# WEBCAM STREAM
# ============================================================

@app.get("/api/v1/streaming/camera_01")
def webcam_stream():

    return StreamingResponse(
        generate_frames(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )