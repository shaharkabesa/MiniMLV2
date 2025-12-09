import numpy as np
import math
from PIL import Image
class neuronmpy:

    def __init__(self,  target_matrix, model_name, learning_rate):
        self.input_matrix = np.array([])
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

        self.mode = "research"
    def createLayers(self, layeramount):
        size = self.input_matrix.shape
        scale = 1.0 / np.sqrt(size[1]) 
        
        new_matrix = np.random.uniform(-0.01, 0.01, (size[1], 10))
        
        print(new_matrix.shape)
        self.weight_matrix = new_matrix
        if self.debug_mode:
            print()
            print("Weight Matrix Populated")
        
    def debugMode(self):
        self.debug_mode = True

    def forwardpass(self):
        print(self.input_matrix)
        logitz = self.input_matrix @  self.weight_matrix

        print(f"Logits result: {logitz}")
        self.logitz_matrix = logitz
        self.softmax()  

    def softmax(self):
        # taking the logitz from forward pass and processing them through softmax equation
        self.probability_matrix = np.exp(self.logitz_matrix) 
  
        # calculating the sum of all P's
        total = np.sum(self.probability_matrix)
        # Creating the P prectange for all arguments
        self.probability_matrix = self.probability_matrix / total
        print(f"Logits result: {self.probability_matrix}")
        if self.mode == "recognize":
            self.getAnswer(self.probability_matrix)
        if self.mode == "research":
            self.lossfunction()

    def lossfunction(self):
       
        self.loss = -np.log(self.probability_matrix @ self.target_matrix[0])
        
        print(f"Current loss: {self.loss[0]}")
        self.calculateError()
    def calculateError(self):
        self.error = self.probability_matrix - self.target_matrix
        print(self.input_matrix.T)    
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
        self.getAnswer(self.probability_matrix)
        self.saveModel()

    def cleanData(self):
        self.loss = 0
        self.logitz_matrix = []
        self.probability_matrix = []
        
        
    def loadimage(self, path_to_image):
        img_data = Image.open(path_to_image).convert("L")
        img_data = np.array(img_data ) 

        if img_data.size > 0:
            print("Data loaded succesfully")
            print(img_data.size)
            self.input_matrix = (img_data / 255.0).reshape(1, -1)
            print(self.input_matrix.shape)
        else:
            print("Invaild path to image")

    
    def saveModel(self): 
        np.save(f"models/{self.model_name}", self.weight_matrix)

    def loadModel(self):
        self.weight_matrix = np.load(f"models/{self.model_name}.npy")

    def getAnswer(self, probablity_matrix):
        answerList = ["A","B", "C", "D", "E", "F", "G", "H", "I", "J"]
        indexWinner = np.argmax(probablity_matrix)
        certainty = np.max(probablity_matrix) * 100
        print(f"The machine predicts its an {answerList[indexWinner]} with  {certainty:0.2f} %")

    def recognize(self):
        self.mode = "recognize"
        self.forwardpass()