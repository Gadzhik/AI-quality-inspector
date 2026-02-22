import cv2
import numpy as np
import random

def create_defective_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Randomly add defects
        if random.random() < 0.1:  # 10% chance of a defect appearing per frame logic is a bit aggressive, maybe per interval
            pass

        # Let's make defects persistent for a few frames or random flicker?
        # User asked for "randomly draws on frames". Let's do it per frame for simplicity or small bursts.
        # To make it look like real objects on a belt, they should move, but random drawing is requested.
        
        if random.random() < 0.3: # 30% of frames have some defect artifact
            defect_type = random.choice(['ellipse', 'polyline'])
            
            if defect_type == 'ellipse':
                center_x = random.randint(0, width)
                center_y = random.randint(0, height)
                axes_x = random.randint(10, 50)
                axes_y = random.randint(10, 50)
                angle = random.randint(0, 360)
                cv2.ellipse(frame, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, (0, 0, 0), -1)
            
            elif defect_type == 'polyline':
                pts = np.array([[random.randint(0, width), random.randint(0, height)] for _ in range(5)], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 0, 0), thickness=3)

        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    create_defective_video("sample1.mp4", "production_with_defects.mp4")
