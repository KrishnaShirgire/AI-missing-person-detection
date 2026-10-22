#  It helps assess whether a detected face is usable, including blur-related quality checks
#  04
import cv2


def check_face_quality(frame, bbox):
    """
    Check whether a detected face is good enough
    for further processing.
    """

    x1, y1, x2, y2 = bbox.astype(int)

    # Make sure coordinates are inside the image
    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Extract face region
    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return False, 0

    # Check face size
    face_width = x2 - x1
    face_height = y2 - y1

    min_face_size = 80

    if face_width < min_face_size or face_height < min_face_size:
        return False, 0

    # Convert face to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Calculate blur using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Minimum sharpness threshold
    blur_threshold = 100

    if blur_score < blur_threshold:
        return False, blur_score

    return True, blur_score