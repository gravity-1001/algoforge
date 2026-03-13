import cv2
import os
import numpy as np

IMAGE_FOLDER = "overlay_results"
OUTPUT_VIDEO = "drone_hud_demo.mp4"

images = sorted(os.listdir(IMAGE_FOLDER))

first = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
height, width, _ = first.shape

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    3,  # slower FPS
    (width, height)
)

for i, img_name in enumerate(images):

    frame = cv2.imread(os.path.join(IMAGE_FOLDER, img_name))
    if frame is None:
        continue

    overlay = frame.copy()

    # -------- Simulated FPS --------
    fps = 6.0

    # -------- Simulated drone telemetry --------
    altitude = 10 + i*0.1
    speed = 4.0

    # -------- obstacle detection (red pixels) --------
    red_pixels = np.sum(
        (frame[:,:,2] > 150) &
        (frame[:,:,1] < 80) &
        (frame[:,:,0] < 80)
    )

    if red_pixels > 2000:
        cv2.putText(
            overlay,
            "WARNING: OBSTACLE DETECTED",
            (30,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,255),
            3
        )

    # -------- HUD text --------
    cv2.putText(
        overlay,
        "AI TERRAIN SEGMENTATION",
        (30,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        overlay,
        f"FPS: {fps}",
        (30,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        overlay,
        f"Altitude: {altitude:.1f} m",
        (width-220,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        overlay,
        f"Speed: {speed:.1f} m/s",
        (width-220,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    # -------- legend --------
    legend = [
        ("Grass", (0,255,0)),
        ("Obstacle", (255,0,0)),
        ("Road", (0,0,255)),
        ("Sky", (255,255,0))
    ]

    start_y = height - 110

    for j,(name,color) in enumerate(legend):

        cv2.rectangle(
            overlay,
            (20,start_y + j*25),
            (40,start_y + 20 + j*25),
            color,
            -1
        )

        cv2.putText(
            overlay,
            name,
            (50,start_y + 15 + j*25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )

    video.write(overlay)

    print("Frame added:", img_name)

video.release()

print("Drone HUD video created:", OUTPUT_VIDEO)