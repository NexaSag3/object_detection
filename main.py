import cv2  # Импортируем OpenCV для работы с изображениями и видеопотоком
import numpy as np  # Импортируем numpy для работы с массивами
import os  # Импортируем os для работы с файловой системой

# Пути к файлам модели YOLOv3: веса, конфигурация и имена классов
weights_path = "yolov3.weights"  # Весы модели
config_path = "yolov3.cfg"  # Конфигурация модели
classes_path = "coco.names"  # Список имен классов для модели

# Загрузка нейронной сети YOLO с весами и конфигурацией
net = cv2.dnn.readNet(weights_path, config_path)

# Чтение имен классов из файла и сохранение их в список classes
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Открытие видеопотока с камеры
# Используйте 0 для встроенной камеры (по умолчанию), или 1 для внешней камеры как у меня, если она подключена
cap = cv2.VideoCapture(1)

# Основной цикл для обработки видеопотока
while True:
    ret, frame = cap.read()  # Чтение кадра
    if not ret:
        print("Не удалось получить кадр")
        break

    # Определяем размеры кадра
    height, width, _ = frame.shape

    # Преобразование кадра в blob для подачи в нейронную сеть
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Получение имен выходных слоев
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Запуск модели и получение предсказаний
    outs = net.forward(output_layers)

    # Инициализация списков для хранения результатов детекции
    class_ids = []
    confidences = []
    boxes = []

    # Обработка каждого предсказания для получения боксов и меток
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Вероятности для каждого класса
            class_id = np.argmax(scores)  # Индекс класса с максимальной вероятностью
            confidence = scores[class_id]  # Значение вероятности

            # Проверка порога вероятности
            if confidence > 0.5:
                # Преобразование координат центра и размеров объекта в координаты бокса
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Вычисление верхнего левого угла бокса
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Сохранение боксов, вероятностей и идентификаторов классов
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применение Non-Maximum Suppression для удаления перекрывающихся боксов
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отображение боксов и меток на кадре
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Имя класса объекта
            color = (0, 255, 0)  # Зеленый цвет для бокса
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    # Отображение кадра с боксов и метками
    cv2.imshow("Image", frame)

    # Выход из программы при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
