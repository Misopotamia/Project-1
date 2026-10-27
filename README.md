# RetailPulse: Dwell-Time & Hot-Spot Analytics

*OpenCV AI Competition 2026, powered by AWS*

RetailPulse is a vision-based analytics system designed to track customer attention, measure dwell times at specific store shelves, and spot high-traffic "hot spots" in physical retail environments.



#Overview

Most retail tracking setups only count foot traffic or monitor queue lines. RetailPulse goes a step further by looking at actual customer engagement—how long people linger in front of specific displays or kiosks.

Using real-time object detection and tracking, the system filters out people who are just walking past or making minor adjustments while standing. It logs exact dwell durations and automatically pushes the metrics to AWS.



#Key Features

*Multi-Object Tracking: Built on Ultralytics YOLOv8s and ByteTrack for reliable, real-time tracking of multiple targets.
*Smart Dwell Logic: Uses motion filtering (`MAX_MOVING_STREAK` and spatial thresholds) to ignore casual passersby and accurately capture stationary engagement, accounting for natural human swaying.
*Hot-Spot Coordinates: Calculates and averages precise physical coordinates (`avg_x`, `avg_y`) where dwell events happen.
*AWS Integration: Syncs behavioral summaries directly to Amazon DynamoDB using boto3.
*Apple Silicon Support: Leverages MPS (Metal Performance Shaders) for fast local processing and testing on macOS.



#Tech Stack

*Language: Python 3.12.0
*Computer Vision & AI: OpenCV, Ultralytics YOLOv8 (`yolov8s.pt`), ByteTrack
*Hardware Acceleration: Apple Silicon (MPS) / NVIDIA CUDA
*Cloud & Backend: AWS (Amazon DynamoDB, `boto3`)



#Project Structure

```text
RetailPulse
├── main.py          # Video processing and tracking logic
├── config.py        # Project thresholds and file paths
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
