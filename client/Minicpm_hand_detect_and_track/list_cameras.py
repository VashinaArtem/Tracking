"""
Simple utility to list all available cameras and test them.
Run this to find your robot camera index.
"""

import cv2

def list_cameras(max_cameras=10):
    """Test camera indices and list available cameras."""
    print("=" * 60)
    print("Camera Detection Utility")
    print("=" * 60)
    print("\nScanning for cameras...\n")
    
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"Camera {i}: FOUND")
                print(f"  Resolution: {width}x{height}")
                
                # Try to get camera name (not always available)
                backend = cap.getBackendName()
                print(f"  Backend: {backend}")
                
                available_cameras.append(i)
                
                # Show preview
                print(f"  Press any key to close preview...")
                cv2.imshow(f'Camera {i} Preview (press any key)', frame)
                cv2.waitKey(2000)  # Show for 2 seconds, or until key press
                cv2.destroyAllWindows()
                print()
            cap.release()
        else:
            # Camera index exists but can't open (might be in use)
            cap.release()
    
    print("=" * 60)
    if available_cameras:
        print(f"Found {len(available_cameras)} camera(s): {available_cameras}")
        print("\nCamera 0 is usually the laptop's built-in webcam.")
        print("Your robot camera is likely one of the higher indices.")
    else:
        print("No cameras found!")
    print("=" * 60)
    
    return available_cameras


if __name__ == "__main__":
    cameras = list_cameras()
    
    if len(cameras) > 1:
        print("\nTo use a specific camera with the tracking client:")
        print(f"  python hand_tracking_client.py --camera <INDEX>")
        print("\nFor example:")
        for cam_id in cameras:
            cam_type = "laptop webcam" if cam_id == 0 else f"external camera {cam_id}"
            print(f"  python hand_tracking_client.py --camera {cam_id}  # Use {cam_type}")
