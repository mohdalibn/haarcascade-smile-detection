
# Libraries required to run the script
import cv2

smilevideo = cv2.VideoCapture('Assets/video2.mp4')
# the following 2 lines sets the resolution to 640 x 480
smilevideo.set(3, 640)
smilevideo.set(4, 480)


trained_smile_data = cv2.CascadeClassifier('HaarcascadeFiles/Smile.xml')
trained_face_data = cv2.CascadeClassifier(
    'HaarcascadeFiles/haarcascade_frontalface_default.xml')


while True:

    read_successful, frame = smilevideo.read()
    # print(read_successfull)

    if read_successful:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coords = trained_face_data.detectMultiScale(gray_frame)

        for cell in face_coords:
            x, y, width, height = cell
            cv2.rectangle(frame, (x, y), (x + width,
                          y + height), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # the last argument is optional
            FaceText = cv2.putText(
                frame, 'Face', (x, y - 10), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            croped_face = frame[y:y+height, x:x+width]

            grayscaled_face = cv2.cvtColor(croped_face, cv2.COLOR_BGRA2GRAY)

            smile_coords = trained_smile_data.detectMultiScale(
                grayscaled_face, scaleFactor=1.7, minNeighbors=20)

            # this nested loop will scan for smiles within the face
            for cell in smile_coords:
                a, b, c, d = cell
                cv2.rectangle(croped_face, (a, b),
                              (a + c, b + d), (0, 255, 0), 2)

                SmileText = cv2.putText(
                    croped_face, 'Smile', (a, b - 10), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        new_frame = cv2.resize(frame, (720, 480))
        cv2.imshow("Video Smile Detection", new_frame)

        key = cv2.waitKey(1)

        if key == 81 or key == 113:
            cv2.destroyAllWindows()
            smilevideo.release()
            break
    else:
        cv2.destroyAllWindows()
        smilevideo.release()
        break
