import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vectorEmbedding import Num2Vec
from models.utils import get_activation_function


class Loss_BetaVAE(nn.Module):


    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = float(beta)
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, x_recon, x, mu, logvar):
        loss_mse = 0.5 * self.mse_loss(x_recon, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_mse, self.beta * loss_kld


class Loss_AE(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, x_recon, x, mu, logvar):
        return 0.5 * self.mse_loss(x_recon, x), torch.zeros((), dtype=x.dtype, device=x.device)


class Loss_NTXent(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        n = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
        target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, target)


class Num2VecAE(Num2Vec):

    def re_parameterize(self, mu, logvar):
        return mu


class Num2VecSiamese(Num2Vec):

    def __init__(self, *args, **kwargs):
        kwargs["train"] = False
        super().__init__(*args, **kwargs)


class MLPEncoderVAE(nn.Module):

    def __init__(self, d_numerical, n_layers, d_token, ffn_dim, ffn_dropout, activation,
                 d_out, train, input_table=True, d_k=100, **_ignored):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_out = d_out
        self.train_ = train
        self.dim_numerical = d_numerical * d_k if input_table else d_numerical
        self.activation_fun = get_activation_function(act_=activation)
        kw = dict(device=self.device, dtype=torch.float64)

        self.input_layer = nn.Linear(self.dim_numerical, d_token, **kw)
        self.hidden = nn.ModuleList()
        for _ in range(n_layers):
            self.hidden.append(nn.ModuleDict({
                "norm": nn.LayerNorm(d_token, **kw),
                "fc0": nn.Linear(d_token, ffn_dim, **kw),
                "fc1": nn.Linear(ffn_dim, d_token, **kw),
                "drop": nn.Dropout(ffn_dropout),
            }))
        self.out_layer_m = nn.Linear(d_token, d_out, **kw)
        self.out_layer_s = nn.Linear(d_token, d_out, **kw)

        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_token, **kw), nn.ReLU(),
            nn.Linear(d_token, ffn_dim, **kw), nn.ReLU(),
            nn.Linear(ffn_dim, self.dim_numerical, **kw))

    def _pad(self, x):
        if x.shape[1] < self.dim_numerical:
            x = F.pad(x, (0, self.dim_numerical - x.shape[1]))
        return x[:, :self.dim_numerical]

    def re_parameterize(self, mu, logvar):
        if self.train_ and self.training:
            std = logvar.mul(0.5).exp()
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        h = self.activation_fun(self.input_layer(self._pad(x)))
        for blk in self.hidden:
            h = blk["drop"](self.activation_fun(blk["fc1"](blk["fc0"](blk["norm"](h)))))
        h = self.activation_fun(h)
        mu = self.out_layer_m(h)
        s = torch.sigmoid(self.out_layer_s(h))
        if self.train_:
            z = self.re_parameterize(mu, s)
            return self.decoder(z), mu, s
        return None, mu, s


class MLPEncoderAE(MLPEncoderVAE):

    def re_parameterize(self, mu, logvar):
        return mu


MODEL_CLASSES = {
    "Num2Vec": Num2Vec,
    "Num2VecAE": Num2VecAE,
    "Num2VecSiamese": Num2VecSiamese,
    "MLPEncoderVAE": MLPEncoderVAE,
    "MLPEncoderAE": MLPEncoderAE,
}


def build_model(class_name, model_info, train):
    cls = MODEL_CLASSES[class_name]
    return cls(d_numerical=model_info["d_numerical"], d_out=model_info["d_out"],
               n_layers=model_info["num_layers"], d_token=model_info["d_token"],
               head_num=model_info["head_num"], attention_dropout=model_info["att_dropout"],
               ffn_dropout=model_info["ffn_dropout"], ffn_dim=model_info["ffn_dim"],
               activation=model_info["activation"], train=train,
               input_table=model_info["ret_table"], d_k=model_info["dim_k"])
