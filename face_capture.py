import cv2
import os

### for capture samples through camera, you can use it to train your own data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def FrontalFaceCapture(window_name, amount_pics, name):

    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(0)

    path_name = BASE_DIR + "/image/" + name
    os.mkdir(path_name)

    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')

    color = (255, 0, 0)

    num = 1
    while cap.isOpened():
        ok, frame = cap.read() 
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

        
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5, minSize = (32, 32))
        
        if len(faces) > 0:          
            for x, y, w, h in faces:  
                
                img_name = "%s/%s.jpg" % (path_name, name + "-" + str(num))

                roi_gray = gray[y: y+h, x: x+w]
    
                cv2.imwrite(img_name, roi_gray,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                num += 1
                if num > (amount_pics):   
                    break

                
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

            
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d/100' % (num),(x + 30, y + 30), font, 1, color, 4)

            
        if num > (amount_pics): break

        
        cv2.imshow(window_name, frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input("Enter your name: ")
    FrontalFaceCapture("get face", 100, name)