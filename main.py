import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QDialog, QVBoxLayout, QWidget
import cv2
import os
import imutils
import numpy as np

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ingrese su nombre")
        self.layout = QVBoxLayout()
        self.lineEdit = QLineEdit()
        self.button = QPushButton("Entrenar modelo", self)
        self.layout.addWidget(self.lineEdit)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        self.button.clicked.connect(self.guardar_nombre)

    def guardar_nombre(self):        
        self.close()        
        captura = CapturarOpciones(self.lineEdit.text()).iniciar_captura()
        if captura == 'CLOSE':
            CapturarOpciones('none').cerrar()
        if self.lineEdit.text() == 'none':
            self.close()
        


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Proyecto biometríco")        
        self.layout = QVBoxLayout()

        self.button = QPushButton("Capturar opciones", self)
        self.button2 = QPushButton("Entrenar modelo", self)
        self.button3 = QPushButton("Iniciar biometrico", self)

        self.button.clicked.connect(self.open_dialog)
        self.button2.clicked.connect(self.entrenar)
        self.button3.clicked.connect(self.probar)

        # self.setCentralWidget(self.button)

        self.layout.addWidget(self.button)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def open_dialog(self):
        dialog = MyDialog()
        if dialog.exec() == QDialog.Accepted:
            value = dialog.lineEdit.text()
            print("Valor ingresado:", value)
    
    def entrenar(self):
        EntrenarImagenes().entrenar_modelo()

    def probar(self):
        ProbarBiometrico().iniciar_identificacion()

class CapturarOpciones():

    def __init__(self, nombre):
        self.nombre = nombre

    def iniciar_captura(self):
        continuar = True

        while continuar == True:
            nombre = self.nombre            
            if nombre == 'none':
                continuar = False

            personName = nombre
            dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
            personPath = dataPath + '/' + personName
            numCamara = 0
            if not os.path.exists(personPath):
                print('Carpeta creada: ',personPath)
                os.makedirs(personPath)
            try:
                cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)  # 0, 1 son los índices de la cámara
                    #cap = cv2.VideoCapture('Video.mp4')
            except Exception as error:
                print('Error con algo de la cámara ' + str(error))
            '''
            # Cuando estamos en colab o jupyter
            !wget --no-check-certificate \
                https://raw.githubusercontent.com/computationalcore/introduction-to-opencv/master/assets/haarcascade_frontalface_default.xml \
                -O haarcascade_frontalface_default.xml
            '''
            faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            count = 0

            while continuar:
                ret, frame = cap.read()
                if ret == False: break
                frame = imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                auxFrame = frame.copy()

                faces = faceClassif.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                    rostro = auxFrame[y:y+h,x:x+w]
                    rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
                    count = count + 1
                cv2.imshow('frame',frame)

                k =  cv2.waitKey(1)
                if k == 27 or count >= 300:                    
                    break
            cap.release()
            cv2.destroyAllWindows()
            return "CLOSE"

    def cerrar(self):        
        cv2.destroyAllWindows()

class EntrenarImagenes():

    def __init__(self):
        self.dataPath = 'imagenes'

    def entrenar_modelo(self):

        self.dataPath
        peopleList = os.listdir(self.dataPath)
        print('Lista de personas: ', peopleList)

        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = self.dataPath + '/' + nameDir
            print('Leyendo las imágenes...')

            for fileName in os.listdir(personPath):
                print('Caras: ', nameDir + '/' + fileName)
                labels.append(label)
                facesData.append(cv2.imread(personPath+'/'+fileName,0))
                # Ver lo que se esta aprendiendo:
                image = cv2.imread(personPath+'/'+fileName,0)
                cv2.imshow('image',image)
                # eso fue lo que aprendio de
                cv2.waitKey(3)
            label = label + 1

        #print('labels= ',labels)
        #print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
        #print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

        # Métodos para entrenar el reconocedor
        #1. face_recognizer = cv2.face.EigenFaceRecognizer_create()
        #2. face_recognizer = cv2.face.FisherFaceRecognizer_create()
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Entrenando el reconocedor de rostros
        print("Entrenando...")
        face_recognizer.train(facesData, np.array(labels))

        # Almacenando el modelo obtenido
        #face_recognizer.write('modeloEigenFace.xml')
        #face_recognizer.write('modeloFisherFace.xml')

        face_recognizer.write('modeloLBPHFace.xml')
        print("¡Modelo almacenado!")

        cv2.destroyAllWindows()

class ProbarBiometrico():

    def __init__(self) -> None:
        pass

    def iniciar_identificacion(self):


        dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
        numCamara = 0
        imagePaths = os.listdir(dataPath)
        print('imagePaths=',imagePaths)

        # 1 #face_recognizer = cv2.face.EigenFaceRecognizer_create()
        # 2 #face_recognizer = cv2.face.FisherFaceRecognizer_create()
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Leyendo el modelo
        #face_recognizer.read('modeloEigenFace.xml')
        #face_recognizer.read('modeloFisherFace.xml')
        face_recognizer.read('modeloLBPHFace.xml')

        cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)
        #cap = cv2.VideoCapture('Video.mp4')

        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

        while True:
            ret,frame = cap.read()
            if ret == False: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()

            faces = faceClassif.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)

                cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
                # LBPHFace
                if result[1] < 70:
                    cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                else:
                    cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1)
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())