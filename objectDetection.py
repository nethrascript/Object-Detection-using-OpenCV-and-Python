import os
import time
from typing import Optional, Tuple

import cv2


def resolve_path(relative_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)


def find_camera(max_index: int = 4) -> Optional[cv2.VideoCapture]:
    for camera_index in range(max_index):
        capture = cv2.VideoCapture(camera_index)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                print(f"Camera found at index {camera_index}")
                return capture
            capture.release()
    return None


def open_video_or_camera() -> Tuple[cv2.VideoCapture, bool]:
    print("Trying to find camera...")
    camera_capture = find_camera()
    if camera_capture is not None:
        return camera_capture, True

    print("No camera found. Falling back to video file.")
    video_file = resolve_path('traffic-mini.mp4')
    if not os.path.exists(video_file):
        raise FileNotFoundError(
            f"Video file not found: {video_file}. Place a video in repo root or plug in a camera."
        )

    file_capture = cv2.VideoCapture(video_file)
    if not file_capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_file}")
    print(f"Using video file: {video_file}")
    return file_capture, False


def configure_capture(capture: cv2.VideoCapture, is_camera: bool, width: int, height: int) -> None:
    if is_camera:
        # Reduce latency and CPU cost on webcams
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def build_detection_model(input_width: int, input_height: int) -> cv2.dnn_DetectionModel:
    config_path = resolve_path('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    weights_path = resolve_path('frozen_inference_graph.pb')
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model files not found. Expected at: {config_path} and {weights_path}"
        )

    model = cv2.dnn_DetectionModel(weights_path, config_path)
    model.setInputSize(input_width, input_height)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    # Backend/target auto-selection for best available speed
    try:
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("DNN: Using CUDA backend/target")
        elif cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("DNN: Using OpenCL target")
        else:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("DNN: Using CPU target")
    except Exception as error:
        # Fall back safely to CPU if anything goes wrong
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"DNN backend selection failed, using CPU. Reason: {error}")

    return model


def load_class_names() -> list:
    names_path = resolve_path('coco.names')
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Class names file not found: {names_path}")
    with open(names_path, 'rt') as file:
        return file.read().rstrip('\n').split('\n')


def main() -> None:
    # Tunables
    display_width, display_height = 640, 480
    input_width, input_height = 320, 320   # Smaller = faster, 320 is model default
    confidence_threshold = 0.5
    nms_threshold = 0.4
    detect_every_n_frames = 1  # Increase to 2-4 to skip frames and boost FPS

    capture, is_camera = open_video_or_camera()
    configure_capture(capture, is_camera, display_width, display_height)

    class_names = load_class_names()
    print(f"Loaded {len(class_names)} class names")

    detection_model = build_detection_model(input_width, input_height)

    print("Press 'q' to quit the application")

    previous_boxes = []
    previous_class_ids = []
    previous_confidences = []

    frame_index = 0
    last_time = time.time()
    fps_text = ""

    while True:
        success, frame = capture.read()
        if not success or frame is None:
            print("End of video or failed to grab frame")
            break

        run_detection = (frame_index % detect_every_n_frames) == 0

        if run_detection:
            class_ids, confidences, boxes = detection_model.detect(
                frame, confThreshold=confidence_threshold, nmsThreshold=nms_threshold
            )

            if len(class_ids) == 0:
                previous_boxes = []
                previous_class_ids = []
                previous_confidences = []
            else:
                previous_boxes = boxes
                previous_class_ids = class_ids.flatten().tolist()
                previous_confidences = confidences.flatten().tolist()

        # Draw the last known detections
        for (box, class_id, conf) in zip(previous_boxes, previous_class_ids, previous_confidences):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = class_names[class_id - 1].upper() if 1 <= class_id <= len(class_names) else str(class_id)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # FPS (instantaneous)
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_text = f"FPS: {1.0 / dt:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

