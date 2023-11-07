import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=7*(20+1), out_features=512)
        self.hidden1 = nn.Linear(in_features=512, out_features=512)
        self.hidden2 = nn.Linear(in_features=512, out_features=512)
        self.hidden3 = nn.Linear(in_features=512, out_features=512)
        self.pi_raw = nn.Linear(in_features=512, out_features=7)
        self.pi_sigma = nn.Linear(in_features=512, out_features=7)
        self.value = nn.Linear(in_features=512, out_features=1)
        
        self.gain = np.sqrt(2.)
        nn.init.orthogonal_(self.input.weight, gain=self.gain)
        nn.init.orthogonal_(self.hidden1.weight, gain=self.gain)
        nn.init.orthogonal_(self.hidden2.weight, gain=self.gain)
        nn.init.orthogonal_(self.hidden3.weight, gain=self.gain)
        nn.init.orthogonal_(self.pi_raw.weight, gain=self.gain)
        nn.init.orthogonal_(self.pi_sigma.weight, gain=self.gain)
        nn.init.orthogonal_(self.value.weight, gain=self.gain)

    def forward(self, obs: torch.Tensor):
        h = F.relu(self.input(obs))
        h = F.relu(self.hidden1(h))
        h = F.relu(self.hidden2(h))
        h = F.relu(self.hidden3(h))
        sigma = torch.clamp(2*self.pi_sigma(h), min=1.0e-4)
        covMat = sigma[:, :, None] * torch.eye(sigma.size(1), device=device)[None, :, :]
        mean = self.pi_raw(h)
        pi = MultivariateNormal(mean, covMat)
        value = -100 * self.value(h).reshape(-1)
        print('mean:', mean[0][0].item(), '\tsigma:',sigma[0][0].item(),'\tvalue:',value[0].item(),)
        return pi, value


if __name__ == "__main__":
    '''
    model = ActorCriticNet(deltaMax=0.1).to(device)
    obs = np.zeros((20+1,7)).reshape(1, -1)
    pi, value = model(torch.tensor(obs, dtype=torch.float32, device=device))
    print("pi", pi, "value", value)
    obs = np.ones((20+1,7)).reshape(1, -1)
    pi, value = model(torch.tensor(obs, dtype=torch.float32, device=device))
    print("pi", pi, "value", value)
    '''
    sigma = torch.tensor([[1., 1., 1., 1., 1., 1., 1.],[2., 2., 2., 2., 2., 2., 2.]], device=device)
    print(sigma.shape)
    covMat = sigma[:,:, None] * torch.eye(sigma.shape[1], device=device)[None,:, :]
    print(sigma)
    print(sigma[:,:,None])
    print(covMat)
    dist = MultivariateNormal(torch.tensor([[1.,2.,3.,4],[5.,6.,7.,8.]], device=device), torch.tensor([[1.],[1.]], device=device))
    print(dist.sample())
