import numpy as np
import math

class neuronmpy:

    def __init__(self, input_matrix, target_matrix, model_name, learning_rate):
        self.input_matrix = input_matrix
        self.target_matrix = target_matrix
        self.weight_matrix = np.array([])
        self.logitz_matrix = np.array([])
        self.model_name = model_name
        self.probability_matrix = np.array([])
        self.debug_mode = False
        self.gradient = []
        self.loss = 0
        self.error = []
        self.learning_rate = learning_rate
    def createLayers(self, layeramount):
        size = self.input_matrix.shape
        new_matrix = np.random.uniform(-1.0, 1.0,(size[1],size[1]))
        
        self.weight_matrix = new_matrix
        if self.debug_mode:
            print("Weight Matrix Populated")
        
    def debugMode(self):
        self.debug_mode = True

    def forwardpass(self):
        
        logitz = self.input_matrix @  self.weight_matrix
        self.logitz_matrix = logitz
        self.softmax()  

    def softmax(self):
        # taking the logitz from forward pass and processing them through softmax equation
        self.probability_matrix = np.exp(self.logitz_matrix) 
  
        # calculating the sum of all P's
        total = np.sum(self.probability_matrix)
        # Creating the P prectange for all arguments
        self.probability_matrix = self.probability_matrix / total

        
        self.lossfunction()

    def lossfunction(self):
        highestprob = np.argmax(self.probability_matrix)
        self.loss = -np.log(self.probability_matrix @ self.target_matrix[0])
        
        print(f"Current loss: {self.loss}")
        self.calculateError()
    def calculateError(self):
        self.error = self.probability_matrix - self.target_matrix    
        self.gradient = self.input_matrix.T @ self.error
        self.calculateWeight() 

    def calculateWeight(self):
        self.weight_matrix = self.weight_matrix - (self.gradient * self.learning_rate)

    def startResearch(self, amount):
        tracker = 0
        for cycle in range(amount):
            self.cleanData()
            self.forwardpass()
            tracker += 1

    def cleanData(self):
        self.loss = 0
        self.logitz_matrix = []
        self.probability_matrix = []
        
        
