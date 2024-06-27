

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable


df = pd.read_csv('../data/merged_daily_avg_weather.csv', index_col = 'Date', parse_dates=True)

label = 'FIT3101'

plt.style.use('ggplot')
# plot the data
plt.figure(figsize=(10, 6))
plt.plot(df[label], label=label)
plt.title('Temperature')
plt.legend()
plt.savefig(f'{label}.png')


X = df.iloc[:, df.columns != label]
y = df[label].values.reshape(-1, 1)


mm = MinMaxScaler()
ss = StandardScaler()


X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y)

train_ratio = 0.7


X_train = X_ss[:int(len(X_ss) * train_ratio), :]
X_test = X_ss[int(len(X_ss) * train_ratio):, :]

y_train = y_mm[:int(len(y_mm) * train_ratio), :]
y_test = y_mm[int(len(y_mm) * train_ratio):, :]


print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

X_train_tensors.shape

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

num_epochs = 10000
learning_rate = 0.005

input_size = X_train_tensors_final.shape[2]
hidden_size = 2
num_layers = 1

num_classes = 1

X_train_tensors_final.shape[1]

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, 30)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final)
  optimizer.zero_grad()

  loss = criterion(outputs, y_train_tensors)

  loss.backward()

  optimizer.step()
  if epoch % 10 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# Đặt mô hình vào chế độ đánh giá (không tính toán gradient)
lstm1.eval()

# Dự đoán trên cả tập huấn luyện và tập kiểm tra
with torch.no_grad():
    train_predict = lstm1(X_train_tensors_final)
    test_predict = lstm1(X_test_tensors_final)

train_predict = train_predict.data.numpy()
test_predict = test_predict.data.numpy()

train_predict = mm.inverse_transform(train_predict)
test_predict = mm.inverse_transform(test_predict)

# train_index = df.index[:350]
# test_index = df.index[350:]

train_index = df.index[:int(len(df) * train_ratio)]
test_index = df.index[int(len(df) * train_ratio):]



# plot train, test prediction and actual data
plt.figure(figsize=(10, 6))
# actual train
plt.plot(train_index, df[label][:int(len(df) * train_ratio)], label='Actual Train')
# actual test
plt.plot(test_index, df[label][int(len(df) * train_ratio):], label='Actual Test')
# predicted train
plt.plot(train_index, train_predict, label='Predicted Train')
# predicted test
plt.plot(test_index, test_predict, label='Predicted Test')
plt.title(f'{label} Prediction')
plt.legend()
plt.savefig(f'{label}_prediction.png')
plt.show()