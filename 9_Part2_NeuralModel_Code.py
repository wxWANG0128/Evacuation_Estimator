import xlrd  
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LinearNET, NeuralNET

##DATASET_PATH
file_path = 'train_2.0.xls' #modify your file path here

##CUDA&MPS_AVAILABLE_CHECK
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    #elif torch.backends.mps.is_available():
     #   return 'mps'
    else:
        return 'cpu'

device = get_device()

##DATA_INITIALISATION
data = xlrd.open_workbook(file_path)
table = data.sheets()[0]
row_n = table.nrows    
col_n = table.ncols

list1 = [[]]*7
for i in range(0,col_n):
    list1[i] = table.col_values(i)

data1 = np.array(list1)

##DATA_STANDARDIZE
data = data1.T

x = data[:,0:6] #features in np.array
y = data[:,6]   #lables in np.array

x = x.T
preprocess = StandardScaler()
x = preprocess.fit_transform(x)
x = x.T
##DATASET_SPLIT
#random_state = random.randint(0,1000)
random_state = 750
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, shuffle = True, random_state = random_state)
#random_state = 750 to obtain the result in report. You can use random_state = random_state if you want.

##DATASET.TO_TENSOR
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

##DATASET.TO_DEVICE
test_features = x_test_tensor.to(device, torch.float64)
test_labels = y_test_tensor.to(device, torch.float64)
train_features = x_train_tensor.to(device, torch.float64)
train_labels = y_train_tensor.to(device, torch.float64)

##FEATURES_NUM
num_inputs = 6

##DATA_LOADER
def data_iter(batch_size, train_features, train_labels):
    num_examples = len(train_features)
    indices = list(range(num_examples))
    random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        j = j.to(device)
        yield  train_features.index_select(0, j), train_labels.index_select(0, j)

'''
##PARAMETER_INITIALIZATION_VINTAGE
w1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_inputs)), dtype = torch.float64, requires_grad = True, device = device)
w2 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype = torch.float64, requires_grad = True, device = device)
b1 = torch.zeros(1, num_inputs, dtype = torch.float64, requires_grad = True, device = device)
b2 = torch.zeros(1, dtype = torch.float64, requires_grad = True, device = device)

##MODEL_DEFINE_VINTAGE
def MODEL_ONE_LAYER(X, w1, b1, w2, b2):
    return torch.mm(torch.relu(torch.mm((X), w1) + b1), w2) + b2
##LOSS_FUNCTION
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2

##OPTIMIZE_FUNCTION_VINTAGE
def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size
'''


lr = 0.0001
#net = LinearNET()
net = NeuralNET()
net.initialize_weights()

lsf = nn.L1Loss()
extra_epochs = 50
batch_size = 30

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

##INITIAL_PARAMETERS
cont = 0
pre_loss = 0
num_epochs = 5000

train_l_list = []
test_valid_list = []
#TRAIN
start_t = time.time()

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, train_features, train_labels):
        #forward
        outputs = net(X)

        #backward
        optimizer.zero_grad()
        loss = lsf(outputs, y)
        loss.backward()

        # update weights
        optimizer.step()

    test_l = lsf(net(test_features),test_labels)
    train_l = lsf(net(train_features), train_labels)
    train_l_list.append(train_l.mean().item())
    test_valid_list.append(test_l.mean().item())
    print('epoch %d, loss %f' % (epoch + 1, test_l.mean().item()))
    if abs(loss-pre_loss) <= 0.00005:
        cont += 1
    else:
        cont = 0
        
    if cont == extra_epochs:
        break
    else:
        pre_loss = loss


end_t = time.time()

print('completed_on:', device)
print('Total time for training:', end_t-start_t)
print('Train_Data_length:', len(train_features))
print('random state number:', random_state)

print ("plot curves")
x1 = range(0, epoch+1)
y1 = train_l_list
y2 = test_valid_list
fig, ax1 = plt.subplots()
ax2 = ax1
lns1 = ax1.plot(x1,y1,'dimgray',label = 'Train Loss')
lns2 = ax2.plot(x1,y2,'silver',label = 'Validation')

ax1.set_xlabel('Epoches')
plt.title('Loss & Validation vs. Epoches')
ax2.set_ylabel('MSE')

lns = lns1 + lns2
lab = [l.get_label() for l in lns]
ax1.legend(lns,lab,loc=0)

plt.show()