import cv2
import numpy as np
import math

# Definir la ruta donde se guardarán las imágenes
save_path = "C:/Users/garci/Documents/visual codigos/codigos Vision Computacional/Examen/"

# Cargar la imagen
img_path = save_path + "camera.png"
img = cv2.imread(img_path)

if img is None:
    print("Error: No se pudo cargar la imagen.")
    exit(1)

# Copia para dibujar detecciones
img_copy = img.copy()

# 1. Convertir a escala de grises y aplicar desenfoque para reducir ruido

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(save_path + "gray.jpg", gray)

# Cargar los clasificadores Haar para rostro y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Detectar rostros en la imagen
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("⚠️ No se detectó ningún rostro. Probando valores más relajados...")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

if len(faces) == 0:
    print("❌ No se detectó ningún rostro después de varios intentos.")
    exit(1)

print(f"✅ {len(faces)} rostro(s) detectado(s).")

# Dibujar rectángulo en la imagen original para mostrar detección
for (x, y, w, h) in faces:
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Verde
cv2.imwrite(save_path + "face_detected.jpg", img_copy)  # Guardar imagen con rostro detectado

# Seleccionar el rostro más grande (por si hay más de uno)
faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
(x, y, w, h) = faces[0]

# Recortar la imagen del rostro
face_roi = gray[y:y+h, x:x+w]

# 2. Ecualizar el histograma SOLO en la región del rostro
face_roi_eq = cv2.equalizeHist(face_roi)
cv2.imwrite(save_path + "cropped_face.jpg", face_roi_eq)


# Intentar detectar ojos en la imagen original antes de la ecualización
eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

if len(eyes) < 2:
    print("⚠️ No se detectaron suficientes ojos en el primer intento. Probando con otro clasificador...")
    eye_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    eyes = eye_cascade_alt.detectMultiScale(face_roi, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10))

if len(eyes) < 2:
    print("⚠️ No se detectaron suficientes ojos después de varios intentos. No se aplicará rotación.")
    angle = 0  # No hay rotación si no hay ojos detectados
else:
    # Ordenar los ojos de izquierda a derecha
    eyes = sorted(eyes, key=lambda e: e[0])
    eye1, eye2 = eyes[:2]

    # Calcular el centro de cada ojo
    eye_center1 = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
    eye_center2 = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)

    # Calcular el ángulo de inclinación
    dy = eye_center2[1] - eye_center1[1]
    dx = eye_center2[0] - eye_center1[0]
    angle = math.degrees(math.atan2(dy, dx))
    print(f"🔄 Ángulo detectado: {angle:.2f} grados")

    # 3. Rotar la imagen original ANTES de recortar el rostro
    (h_img, w_img) = gray.shape[:2]
    center_img = (w_img // 2, h_img // 2)
    M = cv2.getRotationMatrix2D(center_img, angle, 1.0)
    rotated_img = cv2.warpAffine(gray, M, (w_img, h_img), flags=cv2.INTER_CUBIC)

    # Volver a detectar rostros después de la rotación
    faces_rotated = face_cascade.detectMultiScale(rotated_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces_rotated) > 0:
        (x, y, w, h) = faces_rotated[0]
        face_roi_eq = rotated_img[y:y+h, x:x+w]  # Recortar rostro corregido

cv2.imwrite(save_path + "rotated_face.jpg", face_roi_eq)

# 4. Aplicar un filtro de nitidez para mejorar la definición
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharp_face = cv2.filter2D(face_roi_eq, -1, kernel_sharpen)
cv2.imwrite(save_path + "sharp_face.jpg", sharp_face)
cv2.imshow(save_path + "sharp_face.jpg", sharp_face)

print("✅ Procesamiento completado. Imágenes guardadas.")
