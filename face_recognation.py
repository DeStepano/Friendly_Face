from datetime import datetime

import cv2
import face_recognition
import numpy as np
import pandas as pd

# from PIL import Image

main_path = 'just_fr'
tolerance = 0.55
encodings_dir = main_path + '/' + 'encodings'

names_of_latecomers = []
late_time = []

video_capture = cv2.VideoCapture(0)

known_face_names = []
known_face_encodings = []

with open(main_path + '/' + 'known_face_names.txt', 'r') as f:
    for line in f:
        known_face_names.append(str(line.strip()))

for i in range(len(known_face_names)):
    known_face_encodings.append(np.load(encodings_dir + '/' + str(i) + '.npy'))

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)  #!!!
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=5) #num_jitters=5

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name not in names_of_latecomers:
                    names_of_latecomers.append(name)
                    late_time.append(datetime.now().strftime("%d.%m.%Y %H:%M"))

            else:
                pass
                # сохранение неизвестных
                # path = 'D:/OpenCV/Scripts/Images'
                # cv2.imwrite(os.path.join(path, 'waka.jpg'), frame)
                # cv2.imwrite(os.path.join('photos_of_unknowns/' \
                # + datetime.now().strftime("%d.%m.%Y-%H:%M:%S") + '.jpg'), frame)

            face_names.append(name)

    process_this_frame = not process_this_frame  # !!!!

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (60, 20, 235), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (60, 20, 235), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Friendly Face', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        df = pd.DataFrame({'Name': names_of_latecomers, 'Time': late_time})
        df.to_excel('Latecomers Excel/latecomers ' + datetime.now().strftime("%d.%m.%Y") + '.xlsx')
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
