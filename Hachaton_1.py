import cv2
import numpy as np
import torch
from ultralytics import YOLO
import boto3
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
table = dynamodb.Table('RetailPulseAnalytics')

if torch.backends.mps.is_available():
  device = "mps"
elif torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

model = YOLO("yolov8s.pt")
model.to(device)

video_path = "Hard_Example2.mp4"
cap = cv2.VideoCapture(video_path)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
  fps = 30

prev_positions = {}
standing_frames_count = {}
moving_streak = {}
person_locations = {}

current_frame_idx = 0
MIN_WAIT_THRESHOLD = 2.0
MIN_BOX_HEIGHT = 20
MOVEMENT_THRESHOLD = 6
MAX_MOVING_STREAK = 5

skip_frames = 1
cached_render_data = []

print('Starting final smart tracking with AWS... Press "q" to exit.')

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Video ended.")
    break

  current_frame_idx += 1
  is_skipped = current_frame_idx % (skip_frames + 1) != 0

  if not is_skipped:
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.25,
        tracker="bytetrack.yaml",
        verbose=False,
        device=device,
    )

    current_frame_ids = set()
    new_render_data = []

    if results[0].boxes.id is not None:
      boxes = results[0].boxes.xyxy.cpu().numpy()
      track_ids = results[0].boxes.id.cpu().numpy().astype(int)

      for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        box_height = y2 - y1

        if box_height < MIN_BOX_HEIGHT:
          new_render_data.append((x1, y1, x2, y2, track_id, False, 0))
          continue

        current_frame_ids.add(track_id)
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)

        if track_id not in standing_frames_count:
          standing_frames_count[track_id] = 0
        if track_id not in moving_streak:
          moving_streak[track_id] = 0
        if track_id not in person_locations:
          person_locations[track_id] = []

        if track_id in prev_positions:
          prev_x, prev_y = prev_positions[track_id]
          shift = np.sqrt(
              (center_x - prev_x) ** 2 + (center_y - prev_y) ** 2
          )
        else:
          shift = 999

        prev_positions[track_id] = (center_x, center_y)

        if shift < MOVEMENT_THRESHOLD:
          standing_frames_count[track_id] += 1
          moving_streak[track_id] = max(0, moving_streak[track_id] - 1)
          person_locations[track_id].append((center_x, center_y))
        else:
          moving_streak[track_id] += 1
          if moving_streak[track_id] > MAX_MOVING_STREAK:
            standing_frames_count[track_id] = 0
            person_locations[track_id] = []

        wait_time = standing_frames_count[track_id] / fps
        is_active = wait_time >= MIN_WAIT_THRESHOLD

        new_render_data.append(
            (x1, y1, x2, y2, track_id, is_active, int(wait_time))
        )

    disappeared_ids = set(prev_positions.keys()) - current_frame_ids
    for track_id in disappeared_ids:
      total_time = standing_frames_count.get(track_id, 0) / fps
      if total_time >= MIN_WAIT_THRESHOLD:
        locs = person_locations.get(track_id, [])
        if locs:
          avg_x = int(sum([pt[0] for pt in locs]) / len(locs))
          avg_y = int(sum([pt[1] for pt in locs]) / len(locs))

          print(
              f"[SUMMARY] Person ID {track_id} stood for {int(total_time)} sec."
              f" at point (X: {avg_x}, Y: {avg_y})"
          )

          try:
            table.put_item(
                Item={
                    "TrackID": f"ID_{track_id}_{current_frame_idx}",
                    "DwellTimeSeconds": int(total_time),
                    "HotSpotX": int(avg_x),
                    "HotSpotY": int(avg_y),
                }
            )
            print(f"[AWS] Data for ID {track_id} successfully recorded to the cloud!")
          except Exception as e:
            print(f"[AWS Error] Dispatch error: {e}")

      for d in [
          prev_positions,
          standing_frames_count,
          moving_streak,
          person_locations,
      ]:
        if track_id in d:
          del d[track_id]

    cached_render_data = new_render_data

  for x1, y1, x2, y2, track_id, is_active, wait_time in cached_render_data:
    if is_active:
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(
          frame,
          f"ID: {track_id} | Stand: {wait_time}s",
          (x1, y1 - 25),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 0, 255),
          2,
      )
    else:
      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

  cv2.imshow("Smart Retail - Natural Standing", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
