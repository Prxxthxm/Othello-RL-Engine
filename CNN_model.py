import torch
import torch.nn as nn

class CNN_Player(nn.Module):
    def __init__(self,out_channels,conv_size):
        super(CNN_Player,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,out_channels,kernel_size=conv_size,padding=1),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.q_score = nn.Sequential(
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,64)
            # 64 final outputs corresponding to 25 possible moves
        )

    def forward(self,tensor_t): # itensor_t is the input tensor at time step t
      # take in itensor_t
      x = self.features(tensor_t) # extracting features from the tensor
      x = self.flatten(x)
      x = self.q_score(x) # apply the FC layer to get Q(s_t,a_t) value for given state in itensor_t and all possible legal moves a_t
      return x  