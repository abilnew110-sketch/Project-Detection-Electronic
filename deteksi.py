import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


# key = nama kelas COCO, value = label yang ditampilkan
ELEKTRONIK = {
    "watch": "Jam Tangan",
    "tvmonitor": "TV",
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


def deteksi_elektronik(model, frame, confidence):
    hasil = model(frame, conf=confidence, verbose=False)
    deteksi = []

    for kotak in hasil[0].boxes:
        class_id = int(kotak.cls.item())
        nama = model.names[class_id]
        if nama not in ELEKTRONIK:
            continue

        deteksi.append({
            "nama": ELEKTRONIK[nama],
            "confidence": float(kotak.conf.item()),
            "box": kotak.xyxy[0].int().tolist(),
        })

    return deteksi

def main():
    parser = argparse.ArgumentParser(
        description="Deteksi perangkat elektronik menggunakan kamera."
    )
    parser.add_argument("--camera", type=int, default=0, help="Index kamera")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Ambang batas confidence deteksi",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    model = YOLO(str(base_dir / "yolov8n.pt"))
    kamera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not kamera.isOpened():
        print(
            f"Kamera index {args.camera} tidak dapat dibuka. "
            "Coba index lain atau tutup aplikasi lain yang menggunakan kamera."
        )
        return

    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                print("Gagal membaca frame dari kamera.")
                break

            deteksi = deteksi_elektronik(model, frame, args.confidence)

            for item in deteksi:
                x1, y1, x2, y2 = item["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), HIJAU, 2)
                cv2.putText(
                    frame,
                    f"{item['nama']} {item['confidence']:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    HIJAU,
                    2,
                )

            cv2.putText(
                frame,
                f"Elektronik terdeteksi: {len(deteksi)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Deteksi Elektronik", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        kamera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()