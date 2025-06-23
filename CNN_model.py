import torch
import torch.nn as nn

class CNN_Player(nn.Module):
    def __init__(self,out_channels,conv_size):
        super(CNN_Player,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,out_channels,kernel_size=conv_size,padding='same'),
            nn.ReLU(),
        )

        self.q_score = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(out_channels*8*8,128),
            nn.ReLU(),
            nn.Linear(128,64)
            # 64 final outputs corresponding to 25 possible moves
        )

    def forward(self,tensor_t): # itensor_t is the input tensor at time step t
      # take in itensor_t
      x = self.features(tensor_t) # extracting features from the tensor
      x = self.q_score(x) # apply the FC layer to get Q(s_t,a_t) value for given state in itensor_t and all possible legal moves a_t
      return x  