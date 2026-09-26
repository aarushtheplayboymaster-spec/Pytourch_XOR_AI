import torch
import torch.nn as nn

class XOR_AI(nn.Module):
    def __init__(self):
        super().__init__()
        #IT creates the hidden and output layers
        self.hidden = nn.Linear(2,2)
        self.output = nn.Linear(2,1)
    def forward(self,x):
        #IT creates a "forward" faction and changes values into sigmoid values Note(the fuction should be named "forward" and not somthing else!)
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x
#the class XOR_AI is defined as model 
model = XOR_AI()
#mathed for loss formula 
loss_output = nn.MSELoss()
# optimizer is used for changing the weights and biases (this is the same as W = W - learning_rate x N x DS1)
optimizer = torch.optim.SGD(model.parameters(),lr=0.5)
dataset = [
    ([0,0],0),
    ([1,1],1),
    ([1,2],0),
    ([2,1],0),
    ([2,2],1),
    ([1,3],0),
    ([3,1],0),
    ([3,3],1),
    ([1,4],0),
    ([4,1],0),
    ([4,4],1),

]
for loop in range(10000):
    for (input1,input2), targate in dataset:
        x = torch.tensor([[input1,input2]], dtype = torch.float32) #gets the input as x
        R = torch.tensor([[targate]], dtype = torch.float32) #gets the target as R

        optimizer.zero_grad() # resites the blames
        pred = model(x) # actives the model gives INPUTS as I and gets pred
        loss = loss_output(pred,R) # activiates loss fuction 
        loss.backward() # useage loss fuction and backtracks 
        optimizer.step() # changes weights and biases

        #just prints all the results
for (input1,input2), targate in dataset:
    x = torch.tensor([[input1,input2]], dtype = torch.float32) 
    pred = model(x).item()
    print(f'INPUT:{x} | predaction:{pred:.4f} | real output:{targate}')
    