import xlrd  
import torch
import numpy as np 
import random  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

preprocess = StandardScaler()
x = x.T
x = preprocess.fit_transform(x)
x = x.T

##DATASET_SPLIT
random_state = random.randint(0,1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, shuffle = 'true', random_state = 157) 
#random_state = 157 to obtain the result in report. You can use random_state = random_state if you want.

##DATASET.TO_TENSOR
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

##DATASET.TO_DEVICE
test_features = x_test_tensor.to(device, torch.float32)
test_labels = y_test_tensor.to(device, torch.float32)
train_features = x_train_tensor.to(device, torch.float32)
train_labels = y_train_tensor.to(device, torch.float32)

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

##PARAMETER_INITIALIZATION
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype = torch.float32, requires_grad = True, device = device)

##MODEL_DEFINE
def MODEL_ONE_LAYER(X, w):
    return torch.mm(X, w) 

##LOSS_FUNCTION
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2

##OPTIMIZE_FUNCTION
def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size 

##HYPER_PARAMETERS
lr = 0.0005
net = MODEL_ONE_LAYER
loss = squared_loss
extra_epochs = 50
batch_size = 30

##INITIAL_PARAMETERS
cont = 0
pre_loss = 0
num_epochs = 15000

train_l_list = []
test_valid_list = []
#TRAIN
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, train_features, train_labels):
        l = loss(net(X, w), y).sum()
        l.backward()  #BACK_PROBAGATION
        sgd([w], lr, batch_size)  

        ##GRADIENT_CLAER
        w.grad.data.zero_()

    test_l = loss(net(test_features, w), test_labels)
    train_l = loss(net(train_features, w), train_labels)
    train_l_list.append(train_l.mean().item())
    test_valid_list.append(test_l.mean().item())
    print('epoch %d, loss %f' % (epoch + 1, test_l.mean().item()))
    if abs(l-pre_loss) <= 0.00005:
        cont += 1
    else:
        cont = 0
        
    if cont == extra_epochs:
        break
    else:
        pre_loss = l

##OUTPUT
print('\n', w)

print('completed_on:', device)
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