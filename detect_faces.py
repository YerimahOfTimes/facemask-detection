import cv2
import matplotlib.pyplot as plt

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the image
img = cv2.imread("test.jpg")

# Check if image loaded
if img is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Check if cascade loaded
print("Cascade loaded:", not face_cascade.empty())

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30)
)

print("Faces detected:", len(faces))