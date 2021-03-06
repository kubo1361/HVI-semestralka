import torch
import torch.optim as optim
import os
import json
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Agent:
    def __init__(self, model_name, model, gamma=0.99, lr=0.0001, id=0):
        # init vars
        self.model = model
        self.gamma = gamma
        self.lr = lr

        # device - define and cast
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        print('Device: ', self.device, "\nTensor: ", self.tensor)
        
        
        self.model.to(self.device)

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # identification
        self.name = model_name
        self.id = id

        # create folders for models and logs
        self.writer = SummaryWriter('runs/' + self.name + '/' + str(self.id))
        self.model_path = 'models/' + self.name + '/' 
        self.progress_path = 'progress/' + self.name + '/' + str(self.id) + '.json'

        # progress tracking
        self.episode = 0
        self.score = 0
        self.steps = 0

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.progress_path):
            os.makedirs(self.progress_path)

        self.init_model()

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + '/' + str(self.id) + '.pt'))
        except:
            print('Model not found')

    def init_model(self):
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)


    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path + '/' + str(self.id) + '.pt')

    def save_progress(self):
        progress = {'episode' : self.episode, 'steps' :  self.steps, 'score' :  self.score
            , 'lr' : self.lr}
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f)
            f.close()

    def load_progress(self):
        try:
            with open(self.progress_path, 'r') as f:
                progress = eval(f.read())
                f.close()

            self.episode = progress['episode']
            self.score = progress['score']
            self.steps = progress['steps']
        except:
            print('No progress found')


    def train(self):
        pass
        
    def _write_stats(self, stats):
        # write to tensorboard
        for stat in stats:
            self.writer.add_scalar(stat[0], stat[1], self.episode)

    def act(self, observation):
        pass









                