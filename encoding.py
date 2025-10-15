import os
import face_recognition
import numpy as np

main_path = 'just_fr'
photos_dir = main_path + '/' + 'photos'
encodings_dir = main_path + '/' + 'encodings'
num_jitters = 5 #(случайно увеличенное, повернутое, переведенное, перевернутое)

known_face_encodings = []
known_face_names = []

photos_list = next(os.walk(photos_dir))
for i in photos_list[2]:
    try:
        image = face_recognition.load_image_file(photos_dir + '/' + i)
        encoding = face_recognition.face_encodings(image, num_jitters=num_jitters)[0]
        face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(i.split('.')[0])
    except IndexError:
        continue

for i in range(len(known_face_encodings)):
    np.save(encodings_dir+'/'+str(i), known_face_encodings[i])


with open('known_face_names.txt', 'w') as f:
    for s in known_face_names:
        f.write(str(s) +'\n')
