import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT_InputLayer(nn.Module):
  def __init__(self,
               in_channels = 3,
               emb_dim:int=384,
               num_patch_row:int=2,
               image_size:int=224):
    super(ViT_InputLayer, self).__init__()
    self.in_channels = in_channels
    self.emb_dim = emb_dim 
    self.num_patch_row = num_patch_row 
    self.image_size = image_size

    self.num_patch = self.num_patch_row**2 # width * height 正方　なので **2 により計算
    self.patch_size = int(self.image_size // self.num_patch_row) # 上記　正方　が何個あるか

    # 畳み込み
    self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.emb_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
    
    # クラストークン 
    self.cls_token = nn.Parameter(
      torch.randn(1, 1, emb_dim) 
    )

    # Position Embedding
    self.pos_emb = nn.Parameter(
      torch.randn(1, self.num_patch+1, emb_dim) 
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    z_0 = self.patch_emb_layer(x)
    z_0 = z_0.flatten(2)
    z_0 = z_0.transpose(1, 2)
    z_0 = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)
    z_0 = z_0 + self.pos_emb
    return z_0


class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=3, dropout:float=0.01):
        super(MultiHeadSelfAttention, self).__init__() 
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5 # D_hの二乗根。qk^Tを割るための係数

        # 線形層
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout) 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_patch, _ = z.size()

        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        k_T = k.transpose(2, 3)
        dots = (q @ k_T) / self.sqrt_dh
        attn = F.softmax(dots, dim=-1) # 列方向にsoftmax
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, num_patch, self.emb_dim)
        out = self.w_o(out) 
        return out
    
class VitEncoderBlock(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=8, hidden_dim:int=384*4, dropout: float=0.01):
        super(VitEncoderBlock, self).__init__()
        # Layer Normalization 1
        self.ln1 = nn.LayerNorm(emb_dim)
        # MHSA
        self.msa = MultiHeadSelfAttention(
        emb_dim=emb_dim, head=head,
        dropout = dropout,
        )
        # Layer Normalization 2
        self.ln2 = nn.LayerNorm(emb_dim)
        # MLP
        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout),
            #nn.Linear(emb_dim, hidden_dim), 
            #nn.GELU(),
            #nn.Dropout(dropout), 
            #nn.Linear(hidden_dim, emb_dim), 
            #nn.Dropout(dropout),
            #n.Linear(emb_dim, hidden_dim), 
            #nn.GELU(),
            #nn.Dropout(dropout), 
            #nn.Linear(hidden_dim, emb_dim), 
            #nn.Dropout(dropout)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Encoder Blockの前半部分
        out = self.msa(self.ln1(z)) + z
        # Encoder Blockの後半部分 
        out = self.mlp(self.ln2(out)) + out 
        return out

class ViT(nn.Module): 
    def __init__(self, in_channels:int=3, num_classes:int=10, emb_dim:int=384, num_patch_row:int=2, image_size:int=32, num_blocks:int=7, head:int=8, hidden_dim:int=384*4, dropout:float=0.05):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            num_classes: 画像分類のクラス数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 1辺のパッチの数
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定 
            num_blocks: Encoder Blockの数
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ 
            dropout: ドロップアウト率
        """
        super(ViT, self).__init__()
        # Input Layer
        self.input_layer = ViT_InputLayer(
            in_channels, 
            emb_dim, 
            num_patch_row, 
            image_size)

        # Encoder。Encoder Blockの多段。
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout = dropout
            )
            for _ in range(num_blocks)])

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Layer
        out = self.input_layer(x)
        
        # Encoder
        out = self.encoder(out)

        # クラストークンのみ
        cls_token = out[:,0]

        # MLP Head
        pred = self.mlp_head(cls_token)
        return pred