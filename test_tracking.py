# is used to test whether the system can detect a person and assign a tracking ID(for testing)
# 11
import time
from tracking_manager import TrackManager


manager = TrackManager()

# Person appears
manager.update_track(1)

print("Person detected")
print("Track ID:", 1)

time.sleep(2)

# Person appears again
manager.update_track(1)

print("Person detected again")

duration = manager.get_duration(1)

print("Track duration:", round(duration, 2), "seconds")