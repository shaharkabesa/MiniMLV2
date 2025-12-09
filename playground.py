from libs.neuronmpy import neuronmpy
import numpy as np
import random as rnd

input_matrix = np.array([[0.25,0.75]])

input_target = np.array([[0.0, .0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0]])

newmodel = neuronmpy(input_target, "model001", 0.00005)


newmodel.loadimage("imgs/128x128B.jpeg")
# newmodel.debugMode()
newmodel.createLayers(1)


newmodel.loadModel
for i in range(1000):
    x_random = rnd.randrange(0,1)
    if x_random == 1: 
        newmodel.loadimage("imgs/128x128B.jpeg")
    else:
        newmodel.loadimage("imgs/128x128A.jpeg")

        newmodel.startResearch(1)



# newmodel.loadModel()

# newmodel.recognize()