import cv2
import numpy as np
import random
import os

def create_realistic_defective_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Using 'mp4v' or 'XVID' as requested
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print(f"Processing video... Input: {input_path}, Output: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 10% probability of a defect appearing on this frame
        if random.random() < 0.10:
            defect_type = random.choice(['bruise', 'bone'])
            
            if defect_type == 'bruise':
                # Organic dark-red spot (hematoma)
                # Dark red color (B, G, R) -> (20, 20, 100)
                center_x = random.randint(50, width - 50)
                center_y = random.randint(50, height - 50)
                radius = random.randint(15, 40)
                color = (20, 20, 100)
                
                # Draw circle on a copy/overlay
                overlay = frame.copy()
                cv2.circle(overlay, (center_x, center_y), radius, color, -1)
                
                # Blur the overlay to soften edges
                overlay = cv2.GaussianBlur(overlay, (21, 21), 0)
                
                # Blend with original frame
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            elif defect_type == 'bone':
                # Thin polyline, ivory color (B, G, R) -> (210, 240, 255)
                color = (210, 240, 255)
                
                start_x = random.randint(50, width - 50)
                start_y = random.randint(50, height - 50)
                
                points = []
                curr_x, curr_y = start_x, start_y
                for _ in range(random.randint(4, 8)):
                    points.append([curr_x, curr_y])
                    curr_x += random.randint(-10, 10)
                    curr_y += random.randint(-10, 10)
                
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.polylines(frame, [pts], False, color, thickness=2, lineType=cv2.LINE_AA)

        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Done!")

if __name__ == "__main__":
    create_realistic_defective_video("production.mp4", "production_with_realistic_defects.mp4")
