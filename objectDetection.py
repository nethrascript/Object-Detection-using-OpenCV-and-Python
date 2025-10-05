## Detecting Objects

import cv2                                  

# Try different camera indices to find an available camera
def find_camera():
    for i in range(4):  # Try indices 0, 1, 2, 3
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera found at index {i}")
                return cap
            cap.release()
    return None

# Initialize camera or video file
print("Trying to find camera...")
cap = find_camera()

if cap is None:
    print("No camera found! Using video file instead.")
    print("Make sure you have a video file named 'traffic-mini.mp4' in this folder.")
    print("Or change the filename below to match your video file.")
    
    # Try to open video file
    import os
    video_file = 'object-detection/traffic-mini.mp4'  # Updated to correct relative path
    video_path = os.path.abspath(video_file)
    print(f"Attempting to open video file at: {video_path}")
    if not os.path.exists(video_file):
        print(f"File does not exist: {video_path}")
        exit()
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'")
        print("Please place a video file in this folder or fix your camera drivers.")
        print("\nTo fix camera drivers:")
        print("1. Open Device Manager")
        print("2. Find 'HP TrueVision HD Camera' under Cameras")
        print("3. Right-click → Update driver")
        print("4. Or try 'Uninstall device' then restart")
        exit()
    else:
        print(f"Successfully opened video file: {video_file}")

# cap = cv2.VideoCapture('traffic-mini.mp4')

cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = "object-detection/coco.names"

with open(classFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')
print(className)
a = className

configPath = 'object-detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'object-detection/frozen_inference_graph.pb'

## default Configuration
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

print("Press 'q' to quit the application")

while True:
    success, img = cap.read()

    if not success or img is None:
        print("End of video or failed to grab frame")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)
    

    if len(classIds)!=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,255,0), thickness=2)
            print(classId)
            #print(a[classId-1])
            cv2.putText(img, a[classId-1].upper(),(box[0]+10, box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)

    

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):     
        break
    
cap.release()                       
cv2.destroyAllWindows()
    


