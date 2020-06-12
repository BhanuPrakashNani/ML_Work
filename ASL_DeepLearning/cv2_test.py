import numpy as np
import cv2
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image

model = load_model('/home/bhanu/Documents/CNNmodel2.h5')
def prediction(pred):
    return(chr(pred+ 65))

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def main():

	cap = cv2.VideoCapture(0)

	while(True):

	    ret, frame = cap.read()
	    im2 = crop_image(frame, 300,300,300,300)
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    blurred = cv2.GaussianBlur(gray,(15,15),0)
	    im3 = cv2.resize(blurred, (28,28), interpolation = cv2.INTER_AREA)
	    im4 = np.resize(im3, (28, 28, 1))
	    im5 = np.expand_dims(im4,axis=0)
	    pred_probab, pred_class = keras_predict(model, im5)
	    curr = prediction(pred_class)
	    cv2.putText(frame, curr, (350, 350), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
	    cv2.rectangle(frame, (200, 200), (650, 650), (255, 255, 00), 3)
	    cv2.imshow("frame",frame)
	    print(curr)
	    # cv2.imshow('frame',blurred)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
if __name__ == '__main__':
    main()

cap.release()
cv2.destroyAllWindows()
