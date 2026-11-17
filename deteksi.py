from ultralytics import YOLO
import cv2

ELEKTRONIK = {"tv" : "TV", 
              "laptop" : "Laptop", 
              "Handphone" : "cell phone", 
              "mouse" : "Mouse",
              "keyboard" : "Keyboard",
              "remote" : "Remote",
              "refrigator" : "Refrigator",
              }

model = YOLO("yolov8n.pt")
kamera = cv2.VideoCapture(0)

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    hasil = model(frame, conf=0.25, verbose=False)
    print([model.names[int(k.cls[0])] for k in hasil [0].boxes])

    for kotak in hasil[0].boxes:
        nama = model.names[int(kotak.cls[0])]
        if nama not in ELEKTRONIK:
            continue
        yakin = float(kotak.conf[0])
        x1,y1,x2,y2 = map(int, kotak.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{ELEKTRONIK[nama]} {yakin:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()