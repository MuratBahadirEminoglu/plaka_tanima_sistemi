import os
import cv2
import pytesseract
import time

# Tesseract yolunu belirtin
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Hedef klasör
output_folder = "./IMAGES"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Haarcascade yükleme
cascade_path = "C:/Users/murat/Desktop/yapazekaproje/plaka_tanima_sistemi/haarcascade_russian_plate_number.xml"
plateCascade = cv2.CascadeClassifier(cascade_path)

# Kamera ayarları
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Genişlik
cap.set(4, 480)  # Yükseklik
cap.set(10, 150)  # Parlaklık

count = 0  # Kaydedilen plaka sayısını tutar
last_plate_time = 0  # Son plaka yazma zamanı
plate_text = ""  # Plaka metni

while True:
    success, img = cap.read()
    if not success:
        print("Kamera görüntüsü alınamadı!")
        break

    # Görüntüyü gri tonlamaya çevir
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4, minSize=(50, 50))

    # Plakaları algıla ve çerçeve içine al
    for (x, y, w, h) in numberPlates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        imgRoi = img[y:y + h, x:x + w]  # Plakayı kesip al

        # Görüntü ön işleme
        imgRoiGray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
        imgRoiThresh = cv2.threshold(imgRoiGray, 150, 255, cv2.THRESH_BINARY)[1]

        # OCR ile plaka yazısını algıla
        new_plate_text = pytesseract.image_to_string(imgRoiThresh, config='--oem 3 --psm 7')
        new_plate_text = ''.join(e for e in new_plate_text if e.isalnum())  # Sadece harf ve rakamları al

        # Plaka metni her 5 saniyede bir güncellenir
        current_time = time.time()
        if new_plate_text != plate_text and current_time - last_plate_time >= 2:
            plate_text = new_plate_text
            last_plate_time = current_time

        # Plaka yazısını dinamik boyutla çerçevenin üstüne yazdır
        font_scale = max(0.5, min(2, w / 200))
        cv2.putText(img, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        # Algılanan plakayı göster
        cv2.imshow("Plaka", imgRoi)

    # Ana görüntüyü göster
    cv2.imshow("Result", img)

    # Tuş olaylarını kontrol et
    key = cv2.waitKey(1) & 0xFF

    # 'q' tuşu ile çıkış
    if key == ord('q'):
        break

    # 's' tuşu ile kaydetme
    elif key == ord('s'):
        if 'imgRoi' in locals():  # imgRoi tanımlanmış mı kontrol et
            if plate_text:
                cv2.putText(imgRoi, plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            save_path = f"{output_folder}/plate_{count}.jpg"
            cv2.imwrite(save_path, imgRoi)
            print(f"Plaka kaydedildi: {save_path}")

            # Kaydedildi bilgisini göster
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Kaydedildi", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)

            count += 1  # count değerini artır
        else:
            print("Kaydedilecek plaka bulunamadi!")

# Kamera ve pencereleri serbest bırak
cap.release()
cv2.destroyAllWindows()
