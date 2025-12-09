from libs.neuronmpy import neuronmpy
import numpy as np


input_matrix = np.array([[0.25,0.75]])

input_target = np.array([[1.0, 0]])

newmodel = neuronmpy(input_matrix, input_target, "model001")

newmodel.debugMode()
newmodel.createLayers(1)

newmodel.forwardpass()
