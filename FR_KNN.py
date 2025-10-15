import cv2
import pickle
import face_recognition
import pandas as pd
from datetime import datetime

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

names_of_latecomers = []
late_time = []


def predict(x_img, knn_clf=None, model_path=None, num_times_upsample=1, distance_threshold=0.55):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param x_img: изображение для анализа
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param num_times_upsample: количество дискретизации
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Find face locations
    x_face_locations = face_recognition.face_locations(x_img, number_of_times_to_upsample=num_times_upsample)

    # If no faces are found in the image, return an empty result.
    if len(x_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(x_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings),
                                                                               x_face_locations, are_matches)]


if __name__ == "__main__":
    size_of_frame = 0.5  # 0-1
    n_times_upsample = 1
    frame_to_recognition = 2  # Обрабатывается каждый n-ный кадр

    video_capture = cv2.VideoCapture(0)
    number_frame = 0

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=size_of_frame, fy=size_of_frame)  # fx and fy = коэф разрешения
        rgb_small_frame = small_frame[:, :, ::-1]

        if number_frame == frame_to_recognition:  # Обрабатывается каждый n-ный кадр
            number_frame = 0

        if number_frame == 0:
            predictions = predict(rgb_small_frame, model_path="trained_knn_model.clf",
                                  num_times_upsample=n_times_upsample)
            if predictions:
                know_face = True
                for name, coordinates in predictions:
                    print(predictions)
                    if name not in names_of_latecomers and name != 'Unknown':
                        names_of_latecomers.append(name)
                        late_time.append(datetime.now().strftime("%d.%m.%Y %H:%M"))
            else:
                know_face = False
                print('no face!')

        if know_face:
            # Display the results
            for name, (top, right, bottom, left) in predictions:
                top *= int(1/size_of_frame)
                right *= int(1/size_of_frame)
                bottom *= int(1/size_of_frame)
                left *= int(1/size_of_frame)

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 2)  # (60, 20, 235)

                # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (128, 128, 128), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, bottom**0.5/20, (255, 255, 255), 1)

        number_frame += 1

        # Display the resulting image
        cv2.imshow('Friendly Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            df = pd.DataFrame({'Name': names_of_latecomers, 'Time': late_time})
            df.to_excel('Latecomers Excel/latecomers ' + datetime.now().strftime("%d.%m.%Y") + '.xlsx')
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
