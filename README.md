# visionLock 👁️🔒
Computer Vision-Based Smart security System using YOLO and FaceNet

  VisionLock is a smart security access control system powered by computer vision and embedded systems. It combines YOLO for human detection, FaceNet for face verification, and Arduino control to offer secure, intelligent security access and real-time intrusion alerts:
# Activation with Ultrasonic Sensor + Human detection with YOLO
  The system begins by using an ultrasonic sensor to detect the presence of an object near the secured premises. Once an object is detected, the camera is triggered for video capturing and further analysis.<br>
**1.** If the detected object is not a human, the system runs a basic check for two minutes, then shuts down automatically to conserve resources.<br>
**2.** If the object is a human, the system activates the first model—YOLOv8n—which performs real-time person detection to confirm human presence.<br>
# Face Verification with FaceNet
If YOLO confirms a human is present, the image is cropped and passed to FaceNet, which generates a 128D facial embeddings.<br> 
**1.** The embedding is compared against a database of registered faces.<br>
**2.** If a match is found above a certain threshold, the user is authenticated.<br>
# Security Control via Arduino
  Upon successful face verification, a signal is sent over UART to an Arduino which control a solenoid lock to either unlock a door/safe. After 5secs, the door/safe automatically re-locks.<br>
#  Alerting System
If the detected person does not match any registered face in the database (i.e. face verification fails), the system takes the following actions:<br>
**1.** The cropped image of the unverified person is saved locally.<br>
**2.** This image is then uploaded to a cloud-based server (e.g., gofile.io).<br>
**3.** A real-time SMS alert is sent to the authorized user, containing a link to view the uploaded image, so they can visually verify the intruder remotely.<br>

_The entire process runs for 2 minutes, then automatically shuts down and iterate the whole cycle from the beginning if an object is still near it._
**VisionLock stands out as a powerful blend of AI and embedded systems, offering real-time, automated access control with intelligent alerting framework making it an innovative and practical solution for modern home, office security and safe.**


