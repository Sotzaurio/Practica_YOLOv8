from ultralytics import YOLO
import cv2

# Cargar el modelo YOLOv8 preentrenado
model = YOLO("yolov8n.pt")  # Puedes probar con yolov8s.pt si quieres más precisión

# Abrir cámara (0 = cámara por defecto)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    # Ejecutar detección
    results = model(frame)

    # Dibujar resultados
    annotated_frame = results[0].plot()

    # Mostrar ventana
    cv2.imshow("Detección en tiempo real", annotated_frame)

    # Verificar si se detecta un celular
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name == "cell phone":
            print("📱 ¡Celular detectado!")

    # Salir si presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

