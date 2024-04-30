import cv2
import numpy as np
from keras.models import load_model

model= load_model("modelKerass\keras_model.h5", compile = False)
letters = open("modelKerass\labels.txt", "r").readlines()
capture = cv2.VideoCapture(0)
words = ""
addLetter = False
while True:
    success, img = capture.read()
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    findLetter = str(letters[index])
    #letter = str(index)
    letter = findLetter[2:3]
    key = cv2.waitKey(1)
    if key == ord('a'):
         addLetter = True
    elif addLetter:
         words+= letter
         addLetter = False
    print(words)
    cv2.putText(img, words, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(img, letter, (50,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.imshow("Image", img)
                
   
    if key & 0xFF == ord('q'):
         break

capture.release()
cv2.destroyAllWindows()
