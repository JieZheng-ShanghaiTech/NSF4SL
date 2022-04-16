import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_


def xavier_init(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


# Encoder
class MLP(nn.Module):

    def __init__(self, input_size, projection_size, hid_size1=512, hid_size2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hid_size1),
            nn.BatchNorm1d(hid_size1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size1, hid_size2),
            nn.BatchNorm1d(hid_size2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size2, projection_size)
        )

        # self.net.apply(xavier_init)

    def forward(self, x):
        return self.net(x)


# Contrastive module
class Net(nn.Module):

    def __init__(self, input_size, args):
        super(Net, self).__init__()
        self.latent_size = args.latent_size
        self.momentum = args.momentum
        self.input_size = input_size

        self.online_encoder = MLP(input_size, self.latent_size)
        self.target_encoder = MLP(input_size, self.latent_size)

        self.predictor = nn.Linear(self.latent_size, self.latent_size)

        self.online_encoder.apply(xavier_init)
        self.predictor.apply(xavier_init)
        self._init_target()

    def _init_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _update_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, inputs):

        g1, g2, g1_aug, g2_aug = inputs[0], inputs[1], inputs[2], inputs[3]

        g1_online = self.predictor(self.online_encoder(g1_aug))
        g1_target = self.target_encoder(g1)
        g2_online = self.predictor(self.online_encoder(g2_aug))
        g2_target = self.target_encoder(g2)

        return g1_online, g1_target, g2_online, g2_target

    @torch.no_grad()
    def get_embedding(self, inputs):
        # g = self.fuser(inputs)
        g_online = self.online_encoder(inputs.float())
        return self.predictor(g_online), g_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        # Euclidean distance between normalized vectors can be replaced with their negative inner product
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()