import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # TODO: overlay here
        cv2.imshow("Camera_arriere", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        last = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
