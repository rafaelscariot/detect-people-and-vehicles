import cv2
import numpy as np
import urllib.request

IP_CAMERA_ADDRESS = 'http://192.168.0.109:5000/stream'

file_size = (1920, 1080)

RESIZED_DIMENSIONS = (300, 300)
IMG_NORM_RATIO = 0.007843

neural_network = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt',
                                          'models/MobileNetSSD_deploy.caffemodel')

categories = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
              4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
              9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
              13: 'horse', 14: 'motorbike', 15: 'person',
              16: 'pottedplant', 17: 'sheep', 18: 'sofa',
              19: 'train', 20: 'tvmonitor'}

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable",  "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))


def main():
    stream = urllib.request.urlopen(IP_CAMERA_ADDRESS)
    total_bytes = b''

    while True:
        total_bytes += stream.read(1024)
        b = total_bytes.find(b'\xff\xd9')
        print(b)
        if not b == -1:
            a = total_bytes.find(b'\xff\xd8')
            jpg = total_bytes[a:b+2]
            total_bytes = total_bytes[b+2:]

            # decode to colored image ( another option is cv2.IMREAD_GRAYSCALE )
            frame = cv2.imdecode(np.fromstring(
                jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            (h, w) = frame.shape[:2]

            frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS),
                                               IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)

            neural_network.setInput(frame_blob)

            neural_network_output = neural_network.forward()

            for i in np.arange(0, neural_network_output.shape[2]):
                confidence = neural_network_output[0, 0, i, 2]

                if confidence > 0.30:
                    idx = int(neural_network_output[0, 0, i, 1])

                    # detection of bikes, bus, cars, motorbikes and people
                    if idx == 2 or idx == 6 or idx == 7 or idx == 14 or idx == 15:
                        bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                            [w, h, w, h])

                        (startX, startY, endX, endY) = bounding_box.astype("int")

                        label = "{}: {:.2f}%".format(
                            classes[idx], confidence * 100)

                        cv2.rectangle(frame, (startX, startY), (
                            endX, endY), bbox_colors[idx], 2)

                        y = startY - 15 if startY - 15 > 15 else startY + 15

                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, bbox_colors[idx], 2)

            frame = cv2.resize(
                frame, file_size, interpolation=cv2.INTER_NEAREST)

            cv2.imshow('detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    main()
