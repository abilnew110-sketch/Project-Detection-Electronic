from ultralytics import YOLO
import cv2

# key = nama kelas COCO, value = label yang ditampilkan
ELEKTRONIK = {
    "tv": "TV",
    "laptop": "Laptop",
    "cell phone": "Handphone",
    "mouse": "Mouse",
    "keyboard": "Keyboard",
    "remote": "Remote",
    "refrigerator": "Kulkas",
    "microwave": "Microwave",
    "oven": "Oven",
    "toaster": "Toaster",
    "hair drier": "Pengering Rambut",
}

HIJAU = (0, 200, 0)

model = YOLO("yolov8n.pt")
kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not kamera.isOpened():
    raise SystemExit("Coba index 1 atau tutup aplikasi lain yang menggunakan kamera.")

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    hasil = model(frame, conf=0.25, verbose=False)

    jumlah = 0
    for kotak in hasil[0].boxes:
        nama = model.names[int(kotak.cls[0])]
        if nama not in ELEKTRONIK:
            continue

        jumlah += 1
        yakin = float(kotak.conf[0])
        x1, y1, x2, y2 = map(int, kotak.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), HIJAU, 2)
        cv2.putText(frame, f"{ELEKTRONIK[nama]} {yakin:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, HIJAU, 2)

    cv2.putText(frame, f"Elektronik terdeteksi: {jumlah}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()