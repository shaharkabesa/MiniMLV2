from libs.neuronmpy import neuronmpy
import numpy as np
import random as rnd

input_matrix = np.array([[0.25,0.75]])

input_targetA = np.array([[1.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0]])
input_targetB = np.array([[0.0, 1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0]])
input_targetC = np.array([[0.0, 0.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0]])
input_targetD = np.array([[0.0, 0.0 ,0.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0]])
input_targetE = np.array([[0.0, 0.0 ,0.0 ,0.0 ,1.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0]])
input_targetF = np.array([[0.0, 0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,0.0 ,0.0 ,0.0, 0.0]])
input_targetG = np.array([[0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,0.0 ,0.0, 0.0]])
input_targetH = np.array([[0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.0 ,0.0, 0.0]])
input_targetI = np.array([[0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.0, 0.0]])
input_targetJ = np.array([[0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 1.0]])



targets_matrix = np.array([input_targetA,input_targetB,input_targetC,input_targetD,input_targetE,input_targetF,input_targetG,input_targetH,input_targetI,input_targetJ])

newmodel = neuronmpy(input_targetA, "model001", 0.00005)

imageA = newmodel.transformImage("imgs/128x128A.jpeg")
imageB = newmodel.transformImage("imgs/128x128B.jpeg")
imageC = newmodel.transformImage("imgs/128x128C.jpeg")
imageD = newmodel.transformImage("imgs/128x128D.jpeg")
imageE = newmodel.transformImage("imgs/128x128E.jpeg")
imageF = newmodel.transformImage("imgs/128x128F.jpeg")
imageG = newmodel.transformImage("imgs/128x128G.jpeg")
imageH = newmodel.transformImage("imgs/128x128H.jpeg")
imageI = newmodel.transformImage("imgs/128x128I.jpeg")
imageJ = newmodel.transformImage("imgs/128x128J.jpeg")

inputs_matrix = np.array([imageA,imageB,imageC,imageD,imageE,imageF,imageG,imageH,imageI,imageJ])


# # newmodel.debugMode()
# newmodel.createLayers(1)


# newmodel.loadModel
# for i in range(1000):
#     x_random = rnd.randint(0,1)
#     if x_random == 1: 
#         newmodel.target_matrix = input_targetB
#         newmodel.loadimage("imgs/128x128B.jpeg")
#     else:
        
#         newmodel.target_matrix = input_targetA
#         newmodel.loadimage("imgs/128x128A.jpeg")

# newmodel.startResearch(1)
# newmodel.loadModel()
# newmodel.recognize()

# newmodel.multitrain(inputs_matrix,targets_matrix)
newmodel.loadModel()
newmodel.loadimage("imgs/128x128DB.jpeg")
newmodel.recognize()