import numpy as np
from PIL import Image
class neuronmpyv2:

    def __init__(self, model_name):
        self.weights_matrix = []
        self.input_matrix = np.array([])
        self.targets_matrix = np.array([])
        self.weight1 = np.random.uniform(-0.01, 0.01, (16384,128))        
        self.weight2 = np.random.uniform(-0.01, 0.01, (128,10))        
        self.h_matrix = np.array([])
        self.probabilty_matrix = np.array([])
        self.loss = 0
        self.model_name = model_name
        self.mode = "research"
    def addWeight(self, rows, columns):
        
        self.weights_matrix.append(np.random.uniform(-0.01, 0.01, (rows, columns)))
        
    def addInput(self, input_matrix):
        self.input_matrix = input_matrix
    # i need to understand how to convert z = input @ weight 
    def forwardpass(self):
        z1 = self.input_matrix @ self.weight1
        a1 = np.maximum(0,z1)
        self.h_matrix = a1
        z2 = a1 @ self.weight2
        self.probabilty_matrix = z2
        self.softmax()

    def softmax(self):
        self.probabilty_matrix = np.exp(self.probabilty_matrix)
        total = np.sum(self.probabilty_matrix)
        self.probabilty_matrix = self.probabilty_matrix / total
        if self.mode == "research":
            self.calculateloss()
        

    def calculateloss(self):
        print(self.targets_matrix.shape)
        self.loss = -np.log(self.probabilty_matrix @ self.targets_matrix[0])
        print(f"Loss {np.max(self.loss)}")
        self.calculateGradient()

    def calculateGradient(self):
        w2error = self.probabilty_matrix - self.targets_matrix
     
        w2gradient = self.h_matrix.T @ w2error
        
        hidden_error = w2error @ self.weight2.T
        # print(f"Hidden Error PreProcessed: {hidden_error}")
        hidden_error = np.maximum(0, hidden_error)
        # print(f"Hidden Error Processed: {hidden_error}")
        w1gradient = self.input_matrix.T @ hidden_error
        # print(f"Weight 1 Gradient: {w1gradient}") 
        self.calculateWeights(w2gradient, w1gradient)

    def calculateWeights(self,w2grad, w1grad):
        self.weight1 = self.weight1 - (w1grad * 0.10)
        self.weight2 = self.weight2 - (w2grad * 0.10)
        

    def transformImage(self, path_to_image):
        img_data = Image.open(path_to_image).convert("L")
        img_data = np.array(img_data) 

        if img_data.size > 0:
            print("Data loaded succesfully")
            print(img_data.size)
            input_image = (img_data / 255.0).reshape(1, -1)
            print(input_image.shape)
            return input_image
        else:
            print("Invaild path to image")
    def createTarget(self):
        label = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.targets_matrix = np.eye(1,10)
        print(self.targets_matrix)

    def startResearch(self,amount):
        for i in range(amount):
            self.forwardpass()
        self.getAnswer(self.probabilty_matrix)
        self.saveModel()

    def getAnswer(self, probablity_matrix):
        answerList = ["A","B", "C", "D", "E", "F", "G", "H", "I", "J"]
        indexWinner = np.argmax(probablity_matrix)
        certainty = np.max(probablity_matrix) * 100
        print(f"The machine predicts its an {answerList[indexWinner]} with  {certainty:0.2f} %")


    
    def saveModel(self): 
        np.save(f"models/{self.model_name}w1", self.weight1)
        np.save(f"models/{self.model_name}w2", self.weight2)

    def loadModel(self):
        self.weight1 = np.load(f"models/{self.model_name}w1.npy")
        self.weight2 = np.load(f"models/{self.model_name}w2.npy")
        

    def recognize(self):
        self.mode = "recognize"
        self.forwardpass()
        self.getAnswer(self.probabilty_matrix)
    
