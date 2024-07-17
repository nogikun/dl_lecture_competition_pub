import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def torch_log(x):
  return torch.log(torch.clamp(x, min=1e-10))

class LSTM(nn.Module):
  def __init__(self, in_dim, hid_dim):
    super().__init__()
    self.hid_dim = hid_dim
    glorot = 6 / (in_dim + hid_dim * 2)

    self.W_i = nn.Parameter(torch.tensor(np.random.uniform(
      low=-np.sqrt(glorot),
      high=np.sqrt(glorot),
      size=(in_dim + hid_dim, hid_dim)
    ).astype('float32')))
    self.b_i = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

    self.W_f = nn.Parameter(torch.tensor(np.random.uniform(
      low=-np.sqrt(glorot),
      high=np.sqrt(glorot),
      size=(in_dim + hid_dim, hid_dim)
    ).astype('float32')))
    self.b_f = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

    self.W_o = nn.Parameter(torch.tensor(np.random.uniform(
      low=-np.sqrt(glorot),
      high=np.sqrt(glorot),
      size=(in_dim + hid_dim, hid_dim)
    ).astype('float32')))
    self.b_o = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

    self.W_c = nn.Parameter(torch.tensor(np.random.uniform(
      low=-np.sqrt(glorot),
      high=np.sqrt(glorot),
      size=(in_dim + hid_dim, hid_dim)
    ).astype('float32')))
    self.b_c = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

  def function(self, state_c, state_h, x):
    i = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim=1), self.W_i) + self.b_i)
    f = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim=1), self.W_f) + self.b_f)
    o = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim=1), self.W_o) + self.b_o)
    c = f * state_c + i * torch.tanh(torch.matmul(torch.cat([state_h, x], dim=1), self.W_c) + self.b_c)
    h = o * torch.tanh(c)
    return c, h

  def forward(self, x, len_seq_max=0, init_state_c=None, init_state_h=None):
    x = x.transpose(0, 1)  # 系列のバッチ処理のため、次元の順番を「系列、バッチ」の順に入れ替える
    state_c = init_state_c
    state_h = init_state_h
    if init_state_c is None:  # 初期値を設定しない場合は0で初期化する
      state_c = torch.zeros((x[0].size()[0], self.hid_dim)).to(x.device)
    if init_state_h is None:  # 初期値を設定しない場合は0で初期化する
      state_h = torch.zeros((x[0].size()[0], self.hid_dim)).to(x.device)

    size = list(state_h.unsqueeze(0).size())
    size[0] = 0
    output = torch.empty(size, dtype=torch.float).to(x.device)  # 一旦空テンソルを定義して順次出力を追加する

    if len_seq_max == 0:
      len_seq_max = x.size(0)
    for i in range(len_seq_max):
      state_c, state_h = self.function(state_c, state_h, x[i])
      output = torch.cat([output, state_h.unsqueeze(0)])  # 出力系列の追加
    return output
