import time
from collections import defaultdict


class TrackVerification:

    def __init__(self, required_observations=10):

        self.required_observations = required_observations

        self.tracks = defaultdict(
            lambda: {
                "first_seen": time.monotonic(),
                "last_seen": time.monotonic(),
                "observations": [],
                "status": "COLLECTING"
            }
        )


    def update_track(
        self,
        track_id,
        person_id,
        similarity
    ):

        current_time = time.monotonic()

        track = self.tracks[track_id]

        track["last_seen"] = current_time

        track["observations"].append(
            {
                "person_id": person_id,
                "similarity": similarity
            }
        )


    def get_result(self, track_id):

        if track_id not in self.tracks:
            return None

        track = self.tracks[track_id]

        observations = track["observations"]

        if len(observations) == 0:
            return None


        # Count observations for each candidate

        candidate_scores = defaultdict(list)

        for observation in observations:

            candidate = observation["person_id"]

            similarity = observation["similarity"]

            candidate_scores[candidate].append(
                similarity
            )


        # Find candidate with most observations

        best_candidate = max(
            candidate_scores,
            key=lambda candidate:
            len(candidate_scores[candidate])
        )


        scores = candidate_scores[best_candidate]


        average_similarity = sum(scores) / len(scores)


        observation_count = len(observations)


        if observation_count >= self.required_observations:

            status = "POSSIBLE MATCH - HUMAN VERIFICATION REQUIRED"

        else:

            status = "COLLECTING"


        return {
            "track_id": track_id,
            "candidate": best_candidate,
            "observations": observation_count,
            "average_similarity": average_similarity,
            "status": status
        }


    def remove_track(self, track_id):

        if track_id in self.tracks:

            del self.tracks[track_id]