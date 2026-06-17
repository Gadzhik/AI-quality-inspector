import cv2
from .logger import setup_logger

logger = setup_logger(name="CameraStream")

class CameraStream:
    """
    Class to capture video frames synchronously without a background thread.
    """
    def __init__(self, src=0):
        """
        Initialize the camera stream.
        
        Args:
            src: Source index or video file path. Defaults to 0 (webcam).
        """
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera source: {self.src}")
            raise IOError(f"Cannot open camera source: {self.src}")
        
        self.stopped = False
        logger.info(f"Camera stream initialized with source: {self.src}")

    def start(self):
        """
        Included for compatibility with previous calls to .start()
        """
        return self

    def read(self):
        """
        Read a frame synchronously. Loops the video if it reaches the end.
        """
        if self.stopped or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            if isinstance(self.src, str):
                logger.info("Video file ended. Looping back to start.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to loop video")
                    return None
            else:
                logger.warning("Failed to grab frame")
                return None
            
        return frame

    def get_frame_index(self):
        """Индекс кадра, возвращённого последним read().

        cv2 после read() ставит POS_FRAMES на СЛЕДУЮЩИЙ кадр, поэтому текущий =
        pos - 1. Нужно для ground-truth lookup в манифесте дефектов.
        """
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return max(0, pos - 1)

    def stop(self):
        """
        Release resources.
        """
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
        logger.info("Camera stream stopped")

    def __del__(self):
        self.stop()
