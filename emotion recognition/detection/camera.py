import cv2
import torch
from torchvision import transforms
from collections import deque

from model import CNN


predictions = deque(maxlen=10) # to get avr prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)

model = CNN()
model = model.to(device)
weights = torch.load("emotion_model.pth", map_location=device)
model.load_state_dict(weights)
model.eval()

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)


classes = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

while True:
    _, frame = cap.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(img_gray, 1.1, 19)

    for (x, y, w, h) in faces:
        face = img_gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        face = transformer(face)

        face = face.unsqueeze(0)

        face = face.to(device)

        with torch.no_grad():
            output = model(face)

        prediction = output.argmax(dim=1)
        predictions.append(prediction)
        stable_prediction = max(
            set(predictions),
            key=predictions.count
        )

        emotion = classes.get(int(stable_prediction))

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()