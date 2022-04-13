
# Libraries required to run the script
import cv2

trained_smile_data = cv2.CascadeClassifier('HaarcascadeFiles/Smile.xml')
trained_face_data = cv2.CascadeClassifier(
    'HaarcascadeFiles/haarcascade_frontalface_default.xml')

# choose the image to detect the face in
img = cv2.imread('Assets/therock.jpg')


# the next line converts the img to a grayscale image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

# passing the image to the haar cascade algorithm
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


for cell in face_coordinates:  # this code is used to draw rectangles around multiple faces

    x, y, width, height = cell
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    croped_face = img[y:y+height, x:x+width]

    font = cv2.FONT_HERSHEY_SIMPLEX
    # the last argument is optional
    FaceText = cv2.putText(
        img, 'Face', (x, y - 10), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    grayscaled_face = cv2.cvtColor(croped_face, cv2.COLOR_BGRA2GRAY)

    smile_coordinates = trained_smile_data.detectMultiScale(
        grayscaled_face, scaleFactor=1.7, minNeighbors=20)

    # this nested loop will scan for smiles within the face
    for cell in smile_coordinates:
        a, b, c, d = cell
        cv2.rectangle(croped_face, (a, b), (a + c, b + d), (0, 0, 255), 2)

        SmileText = cv2.putText(
            croped_face, 'Smile', (a, b - 10), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

# the following line prompts the display
new_img = cv2.resize(img, (720, 480))  # resized the size of the window
cv2.imshow("Image Smile Detection", new_img)


key = cv2.waitKey()

if key == 81 or key == 113:
    cv2.destroyAllWindows()
