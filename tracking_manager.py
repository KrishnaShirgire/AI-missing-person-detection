# to store track information for each detected person
import time


class TrackManager:

    def __init__(self): # createe a dictionary to store track information
        self.tracks = {}

    def update_track(self, track_id): # update the track information for a given track_id
        current_time = time.time()

        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "first_seen": current_time,
                "last_seen": current_time,
                "face_frames": 0,
                "good_face_frames": 0
            }
        else:
            self.tracks[track_id]["last_seen"] = current_time

    def get_duration(self, track_id): # get the duration of a track in seconds
        if track_id not in self.tracks:
            return 0

        return (
            self.tracks[track_id]["last_seen"]
            - self.tracks[track_id]["first_seen"]
        )

    def get_track(self, track_id): # get the track information for a given track_id
        return self.tracks.get(track_id)