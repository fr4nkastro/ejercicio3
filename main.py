# Import the necessary packages
from consolemenu import *
from consolemenu.items import *
import cv2
import random
import numpy as np

function_linea= 'Function Linea'
function_linea_gruesa='Function Linea Gruesa'
function_linea_delgada='Function Linea Delgada'
function_rectangulo='Function Rectángulo'
function_circulo='Function Círculo'

# funciones
#############################################################################################
def detectionFace(option):
    trained_face_data= cv2.CascadeClassifier('cascadeClassifier.xml')
    path1='videos/Personas.mp4'
    path2='videos/Animales.mp4'
    path3='videos/Animado2.mp4'

    if option==1:
        #Capturar desde video
        captureFace= cv2.VideoCapture(path1)
    elif option==2:
        captureFace= cv2.VideoCapture(path2)
    elif option==3:
        captureFace= cv2.VideoCapture(path3)
    else:
        #Capturar desde camara
        captureFace =cv2.VideoCapture(0)

    def r():
        return random.randrange(256)


    #Caputurar cada frame
    while True:
        succesful_frame, frame= captureFace.read()
        #Frame de colores a escala de grises
        gray_scale_frame=   cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #usar el algoritmo de detección de rostros
        face_coordenate = trained_face_data.detectMultiScale(gray_scale_frame)
        #crear un cuadro alrededor del rostro
        for(x,y,w,h) in face_coordenate:
            #dibujar el recuadro
            # cv2.rectangle(frame, (x,y), (x+w, y+h), (r(),r(),r()))
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,256), thickness=1)
            #Mostrar Los Rostros
            cv2.imshow('Rostros',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    

#############################################################################################
def lineDetection():
    img = cv2.imread('images/lines.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,120)
    minLineLength = 20
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    # print(lines[0])
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("edges", edges)
        cv2.imshow("lines", img)
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#############################################################################################
def detectionCircles():
    path='images/planets.jpg'
    #cargar imagen
    circles_img=cv2.imread(path)

    #convertir img a escala de grises
    img_gray= cv2.cvtColor(circles_img, cv2.COLOR_BGR2GRAY)

    #aplicar el filtro
    img= cv2.medianBlur(img_gray,7)

    #obtener los circulos
    circles= cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,120, param1=100, param2=30, minRadius=0,maxRadius=0)

    #recorrer los circulos
    detected_circles= np.uint(np.around(circles))
    # print(f"Circulos detectados: {detected_circles}")

    for i in detected_circles[0,:]:
        #dibujar las circunferencias
        cv2.circle(circles_img,(i[0], i[1]), i[2], (0,255,0), 2)
        #dibujar el centro
        cv2.circle(circles_img,(i[0],i[1]), 2,(0,0,255),3)
    

    cv2.imshow(path, circles_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#############################################################################################

def drawRectangle(thickness):
    def r():
        return random.randrange(256)
    img = cv2.imread('images/white.jpg')
    cv2.rectangle(img, (100,100), (200, 200), (r(),r(),r()), thickness)
    cv2.imshow('Rectangle',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#############################################################################################
# def detectionRectangle():
#     img = np.zeros((200, 200), dtype=np.uint8)
#     img[50:150, 50:150] = 255
#     ret, thresh = cv2.threshold(img, 127, 255, 0)
#     image, contours, hierarchy = cv2.findContours(thresh, 
#         cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
#     cv2.imshow("contours", color)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

#############################################################################################
def detectionRectangleContours():
    org=cv2.imread('images/foxWhiteBackground.png')
    # org=cv2.imread('images/whiteBackground.png')
    img = cv2.imread('images/foxWhiteBackground.png',0)
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.rectangle(org,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("image", org)
    cv2.waitKey()
#############################################################################################

def function(var):
    print(var)
    input('press any key')

#############################################################################################    


# Create the menu
menu = ConsoleMenu("Computer Vision", "OpenCV")
menu_color= ConsoleMenu("Computer Vision", "OpenCV->Color")
menu_video=ConsoleMenu("Computer Vision",'OpenCV->Video')

# Create some items


# A FunctionItem runs a Python function when selected
# MenuItem is the base class for all items, it doesn't do anything when selected
# A SubmenuItem lets you add a menu (the selection_menu above, for example)
# as a submenu of another menu

menu_linea = FunctionItem("Línea", lineDetection, [])
submenu_linea_gruesa= FunctionItem("Línea Gruesa", drawRectangle, [20])
submenu_linea_delgada= FunctionItem("Línea Delgada",drawRectangle, [2])
menu_color.append_item(submenu_linea_gruesa)
menu_color.append_item(submenu_linea_delgada)
submenu_color= SubmenuItem("Color (RGB)",menu_color, menu)
menu_rectangulo= FunctionItem("Rectángulo",detectionRectangleContours,[])
menu_circulo=FunctionItem("Círculo",detectionCircles, [])
submenu_video=SubmenuItem('Video', menu_video,menu)
menu_video_personas=FunctionItem('Real Con Personas',detectionFace, [1])
menu_video_animado=FunctionItem('Animado (Comics)', detectionFace, [3])
menu_video_animales=FunctionItem('Real Con Animales',detectionFace, [2])
menu_video.append_item(menu_video_personas)
menu_video.append_item(menu_video_animado)
menu_video.append_item(menu_video_animales)





# Once we're done creating them, we just add the items to the menu
menu.append_item(menu_linea)
menu.append_item(submenu_color)
menu.append_item(menu_rectangulo)
menu.append_item(menu_circulo)
menu.append_item(submenu_video)



# Finally, we call show to show the menu and allow the user to interact
menu.show()

