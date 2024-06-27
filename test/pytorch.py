import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

class LSTMPredictor:
    def __init__(self, csv_file, label, sequence_length=30, train_ratio=0.7):
        self.use_cols = ['SDATE', 'TEMP_°C', 'pH', 'TSS_mg_L',
       'COND_mS_m', 'Ca_meq_L', 'Mg_meq_L', 'Na_meq_L', 'K_meq_L', 'ALK_meq_L',
       'Cl_meq_L', 'SO4_meq_L', 'NO32_mg_L', 'NH4N_mg_L', 'TOTP_mg_L',
       'DO_mg_L', 'CODMN_mg_L']
        self.df = pd.read_csv(csv_file, index_col='SDATE', parse_dates=True, usecols=self.use_cols)
        self.label = label
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.ss = StandardScaler()
        self.mm = MinMaxScaler()
        self.model = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device


    def preprocess_data(self):
        X = self.df.drop(columns=[self.label])
        y = self.df[self.label].values.reshape(-1, 1)
        # X shape: (8703, 32)
        # y shape: (8703, 1)

        X_ss = self.ss.fit_transform(X)
        y_mm = self.mm.fit_transform(y)

        X_train = X_ss[:int(len(X_ss) * self.train_ratio), :]
        X_test = X_ss[int(len(X_ss) * self.train_ratio):, :]

        y_train = y_mm[:int(len(y_mm) * self.train_ratio), :]
        y_test = y_mm[int(len(y_mm) * self.train_ratio):, :]

        self.X_train_tensors = Variable(torch.Tensor(X_train))
        self.X_test_tensors = Variable(torch.Tensor(X_test))

        self.y_train_tensors = Variable(torch.Tensor(y_train))
        self.y_test_tensors = Variable(torch.Tensor(y_test))

        self.X_train_tensors_final = torch.reshape(self.X_train_tensors, (self.X_train_tensors.shape[0], 1, self.X_train_tensors.shape[1]))
        self.X_test_tensors_final = torch.reshape(self.X_test_tensors, (self.X_test_tensors.shape[0], 1, self.X_test_tensors.shape[1]))

    def build_model(self):
        input_size = self.X_train_tensors_final.shape[2]
        hidden_size = 2
        num_layers = 1
        num_classes = 1

        self.model = LSTM1(num_classes, input_size, hidden_size, num_layers, self.sequence_length)

    # def train_model(self, num_epochs=10000, learning_rate=0.005):
    #     criterion = torch.nn.MSELoss()
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    #     for epoch in range(num_epochs):
    #         outputs = self.model.forward(self.X_train_tensors_final)
    #         optimizer.zero_grad()

    #         loss = criterion(outputs, self.y_train_tensors)
    #         loss.backward()

    #         optimizer.step()
    #         if epoch % 1000 == 0:
    #             print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    def train_model(self, num_epochs=200, learning_rate=0.005):
        # Check if GPU is available and set the device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        # Move the model to the specified device
        self.model.to(device)

        # Define the loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        loss_values = []

        for epoch in range(num_epochs):
            # Ensure data is on the correct device
            inputs = self.X_train_tensors_final.to(device)
            labels = self.y_train_tensors.to(device)

            # Forward pass: Compute predicted y by passing x to the model
            outputs = self.model.forward(inputs)

            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            loss_values.append(loss.item())

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, loss: {loss.item():1.5f}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(num_epochs), loss_values, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('./figure/loss_curve.png')
        plt.show()

    def score(self, X, y):
        """
        return train score and test score
        """
        train_predict = self.model(X.to(self.device))
        data_predict = train_predict.cpu().data.numpy()
        dataY_plot = y.data.numpy()

        data_predict = self.mm.inverse_transform(data_predict)
        dataY_plot = self.mm.inverse_transform(dataY_plot)

        mae = mean_absolute_error(dataY_plot, data_predict)
        mse = mean_squared_error(dataY_plot, data_predict)
        rmse = np.sqrt(mean_squared_error(dataY_plot, data_predict))
        r2 = r2_score(dataY_plot, data_predict)

        return mae, mse, rmse, r2

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            # print device of model and data
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Data device: {self.X_train_tensors_final.device}")
            train_predict = self.model(self.X_train_tensors_final.to(self.device))
            test_predict = self.model(self.X_test_tensors_final.to(self.device))

        train_predict = train_predict.cpu().data.numpy()
        test_predict = test_predict.cpu().data.numpy()

        train_predict = self.mm.inverse_transform(train_predict)
        test_predict = self.mm.inverse_transform(test_predict)

        train_index = self.df.index[:int(len(self.df) * self.train_ratio)]
        test_index = self.df.index[int(len(self.df) * self.train_ratio):]

        train_score = self.score(self.X_train_tensors_final, self.y_train_tensors)
        test_score = self.score(self.X_test_tensors_final, self.y_test_tensors)
        print(f"Train Score: MAE={train_score[0]}, MSE={train_score[1]}, RMSE={train_score[2]}, R2={train_score[3]}")
        print(f"Test Score: MAE={test_score[0]}, MSE={test_score[1]}, RMSE={test_score[2]}, R2={test_score[3]}")

        if not os.path.exists('./results'):
            os.makedirs('./results')
        with open(f'./results/{self.label}_result.txt', 'w') as f:
            f.write('Train Score: MAE=%f, MSE=%f, RMSE=%f, R2=%f\n' % train_score)
            f.write('Test Score: MAE=%f, MSE=%f, RMSE=%f, R2=%f\n' % test_score)
            f.close()


        # plot train, test prediction and actual data
        plt.figure(figsize=(14, 6))
        plt.plot(train_index, self.df[self.label][:int(len(self.df) * self.train_ratio)], label='Actual Train')
        plt.plot(test_index, self.df[self.label][int(len(self.df) * self.train_ratio):], label='Actual Test')
        plt.plot(train_index, train_predict, label='Predicted Train')
        plt.plot(test_index, test_predict, label='Predicted Test')
        plt.title(f'{self.label} Prediction')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'./figure/{self.label}_prediction.png')
        plt.show()

    def save_model(self, model_file):
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, model_file)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved successfully to {model_path}")

    def load_model(self, model_file):
        self.preprocess_data()
        self.build_model()
        model_path = os.path.join("model", model_file)
        self.model.load_state_dict(torch.load(model_path))
        print(f"Model loaded successfully from {model_file}")

    def predict_future(self, num_samples=30):
        self.model.eval()
        with torch.no_grad():
            future_data = self.X_test_tensors_final[-1, :, :]   # last sequence
            future_data = future_data.view(1, 1, future_data.shape[1])  # reshape to (1, 1, input_size)
            future_prediction = []
            for i in range(num_samples):
                future_predict = self.model(future_data)
                # future_data = torch.cat((future_data[:, :, 1:], future_predict), axis=2)
                future_predict = future_predict.unsqueeze(0)


                future_data = torch.cat((future_data[:, :, 1:], future_predict), dim=2)
                future_prediction.append(future_predict.item())

            future_prediction = self.mm.inverse_transform(np.array(future_prediction).reshape(-1, 1))

            # future datetime theo từng ngày
            # future_datetime = pd.date_range(start=self.df.index[-1], periods=num_samples+1, freq='D')[1:]

            # future datetime theo từng 15 phút
            # future_datetime = pd.date_range(start=self.df.index[-1], periods=num_samples+1, freq='15T')[1:]

            # future datetime theo từng 12 giờ
            future_datetime = pd.date_range(start=self.df.index[-1], periods=num_samples+1, freq='12h')[1:]

            # plot future prediction
            plt.figure(figsize=(14, 6))
            # plt.plot(self.df.index, self.df[self.label], label='Actual Data')
            plt.plot(future_datetime, future_prediction, label='Future Prediction')
            plt.title(f'{self.label} Future Prediction')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'./figure/{self.label}_future_prediction.png')
            plt.show()



class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        device = x.device
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

def parse_arguments():
    parser = argparse.ArgumentParser(description='LSTM Predictor')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--label', type=str, required=True, help='Name of the column to predict')
    parser.add_argument('--sequence_length', type=int, default=30, help='Length of input sequences')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data to total data')
    parser.add_argument('--model_file', type=str, default='lstm_model.pth', help='Path to save/load the model')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--load_model', action='store_true', help='Load a pre-trained model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    lstm_predictor = LSTMPredictor(args.csv, args.label, args.sequence_length, args.train_ratio)

    if args.load_model:
        lstm_predictor.load_model(args.model_file)
        # future prediction
        lstm_predictor.predict_future(num_samples=50)
    else:
        lstm_predictor.preprocess_data()
        lstm_predictor.build_model()
        lstm_predictor.train_model()
        if args.save_model:
            lstm_predictor.save_model(args.model_file)

    lstm_predictor.evaluate_model()
