from libs.neuronmpyv2 import neuronmpyv2
import numpy as np

newp = neuronmpyv2("mlpmodel1")
image_target = newp.transformImage("imgs/128x128A.jpeg")
newp.addInput(image_target)
newp.createTarget()

newp.targets_matrix = input_targetA = np.array([[1.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0]])
# newp.startResearch(100)
newp.loadModel()
newp.recognize()