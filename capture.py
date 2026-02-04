from PIL import Image
import cv2
from pypylon import pylon
import numpy as np
import os

# Initialize the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Start grabbing images
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Image format converter
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Directory to save images
f = r"C:\Users\prart\indusmn\images\4"

# Create directory if it does not exist
if not os.path.exists(f):
    os.makedirs(f)

j = 100

try:
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Convert image to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()

            # Display the image
            cv2.imshow('live_feed', frame)

            # Save the image
            image_path = os.path.join(f, f"r1{j}.jpg")
            success = cv2.imwrite(image_path, frame)
            if success:
                print(f"Image saved successfully: {image_path}")
            else:
                print(f"Failed to save image: {image_path}")
            j += 1

            # Exit on keypress 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Grab failed with error code: {grabResult.ErrorCode}")
        grabResult.Release()
finally:
    # Release the camera and close windows
    camera.StopGrabbing()
    cv2.destroyAllWindows()
    print("Camera stopped and windows closed")
