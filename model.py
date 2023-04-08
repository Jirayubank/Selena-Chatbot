import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # nn.Linear(in_feature, out_feature)
        # weight = (out_feature, in_feature)
        # bias = (out_feature)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)            # input layer
        out = self.relu(out)        # feed to next layer
        out = self.l2(out)          # hidden layer 1
        out = self.relu(out)        # feed to next layer
        out = self.l3(out)          # hidden layer 2
        # no activation and no softmax at the end
        return out                  # output layer
