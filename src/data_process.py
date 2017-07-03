import numpy as np
import sys
import cv2
import os, os.path

#multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

# print("All images have been processed!!!")
# cv2.destroyAllWindows()
cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 4:
        print("Usage: python data_process.py <path/to/cascade/models/> <path/to/input/data/> <path/to/output/data/>")
        quit()
    model_path = sys.argv[1]
    raw_data_path = sys.argv[2]
    output_data = sys.argv[3]
    # Read in OpenCV face and eye cascade model
    face_cascade = cv2.CascadeClassifier(os.path.join(model_path, 'haarcascade_frontalface_alt2.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(model_path, 'haarcascade_eye.xml'))

    # Read in data tag for different people, assume each person has unique name
    # Data for different people are placed in different folders
    #   eg. raw_data/Arthur/example1.png
    #       raw_data/Arthur/example2.png
    #       raw_data/Jason/example3.png
    #       ...
    tags = [tag for tag in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, tag))]
    tag_counter = 0
    labels = []
    filenames = []
    for tag in tags:
        img_counter = 0
        pics = os.listdir(os.path.join(raw_data_path, tag))
        for pic_name in pics:
            img = cv2.imread(os.path.join(raw_data_path, tag, pic_name))
            height = img.shape[0]
            width = img.shape[1]
            size = height * width
            if size > (500^2):
                # print("Resizing ...")
                r = 500.0 / img.shape[1]
                dim = (500, int(img.shape[0] * r))
                img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                img = img2

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            #TODO detect left and right eye in the faces
            eyesn = 0
            if len(faces) > 0:
                #Assuming one face in one image
                (x,y,w,h) = faces[0]

                #TODO: Resize the image here
                imgCrop = gray[y:y+170,x:x+170]

                # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    # print("Eye {0} position, ({1}, {2})".format(eyesn, ex, ey))
                    eyesn = eyesn +1
                if eyesn >= 2:
                    cv2.imwrite(os.path.join(output_data, tag + str(img_counter) + ".png"), imgCrop)
                    #Add to lists, which will be used later for creating csv files
                    labels.append(tag_counter)
                    filenames.append(tag + str(img_counter) + ".png")
                    cv2.imshow('img',imgCrop)
                    print("Image"+str(tag + str(img_counter))+" has been processed and cropped")
                img_counter += 1

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        tag_counter += 1


if __name__ == '__main__':
    main()