import torch as th
from torch import nn

class DuelingDqnNet(nn.Module):
    def __init__(self, obs_shape_dict, num_outputs, local_map_size = (5, 5)):
        super(DuelingDqnNet, self).__init__()
        extractors = {}
        total_concat_size = 0
        self.local_map_size = local_map_size
        for key, subspace in obs_shape_dict.items():
            if key == "image":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(1,32, kernel_size=3),
                    nn.Tanh(),
                    nn.Conv2d(32, 64, kernel_size=3),
                    nn.Tanh(),
                    nn.Flatten(),
                    nn.Linear(64, 128),
                    nn.Tanh(),
                    )
                total_concat_size += 128

                
        self.extractors = nn.ModuleDict(extractors)
        self.position_embedding = nn.Embedding(50*50, 128)
        total_concat_size += 128
        # Update the features dim manually
        # self._features_dim = total_concat_size
        # self.hidden_size = total_concat_size
        # self.lstm_layer = lstm_layer
        # self.rnn = nn.LSTM(input_size=total_concat_size, hidden_size=self.hidden_size, num_layers=self.lstm_layer, batch_first=False)
        self.value = nn.Sequential(
            nn.Linear(total_concat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )
        
        self.advantage = nn.Sequential(
            nn.Linear(total_concat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
            )
    
    def forward(self, observations) -> th.Tensor:
        '''
        input: observations[Dict[str, th.Tensor]]
        image: th.Tensor
        site: (x, y)
        output: th.Tensor
        '''
        import time
        # site = observations['site'].squeeze(1).long()
        image = observations['image'].reshape(-1, 1, 5, 5)
        # # normalize the image
        # image = image.float() / th.abs(image).max()
        # # padding the image to avoid the edge effect
        # image = th.nn.functional.pad(image, (self.local_map_size[0]//2, self.local_map_size[0]//2, self.local_map_size[1]//2, self.local_map_size[1]//2), mode='constant', value=1)
        # # extract the local map
        # local_map = [image_[site_[0]:site_[0]+self.local_map_size[0], site_[1]:site_[1]+self.local_map_size[1]] for image_, site_ in zip(image, site)]
        # local_map = th.stack(local_map).unsqueeze(1)
        # local_map = image[site[:, 0]:site[:, 0]+self.local_map_size[0], site[:, 1]:site[:, 1]+self.local_map_size[1]].unsqueeze(0).unsqueeze(0)
        # extract the features
        site = observations['site'].squeeze(1).long()
        position_embedding = self.position_embedding(site[..., 0]*50 + site[..., 1])
        features = self.extractors['image'](image)
        features = th.cat([features, position_embedding], dim=-1)
        value = self.value(features)
        advantage = self.advantage(features) 
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
class position_embedding(nn.Module):
    def __init__(self, d_model, max_len=50*50):
        super(position_embedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
    def forward(self, x):   
        position = th.arange(0, self.max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, self.d_model, 2) * -(th.log(th.tensor(10000.0)) / self.d_model))
        pe = th.zeros(self.max_len, self.d_model)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        return pe