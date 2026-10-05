import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ASDiffusionBackbone(nn.Module):
    def __init__(self, input_dim, num_classes, real_num_classes, num_types, addition_idx, device, bg_w=2.0, adapter_mode="identity", adapter_scales=(1, 2, 4, 8)):
        super(ASDiffusionBackbone, self).__init__()
        

        self.device = device
        self.addition_idx = addition_idx
        self.num_classes = num_classes
        self.num_types = num_types #4 # normal, modification, slip, correction
        self.prototype_dim = input_dim
        self.real_num_classes = real_num_classes
        # self.rescale_factor = 10
        action_weights = []
        for i in range(num_classes):
            if i == 0 or i >= real_num_classes:
                action_weights.append(bg_w)
            else:
                action_weights.append(1.0)
        self.action_ce = nn.CrossEntropyLoss(weight=torch.tensor(action_weights))
        self.cosine_similarity = nn.CosineSimilarity()
        self.mse = nn.MSELoss(reduction='none')
        num_f_maps = 128
        num_layers = 8
        kernel_size = 5
        dropout_rate = 0.1

        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)
        self.module = MixedConvAttModuleV2(num_layers, num_f_maps, input_dim, kernel_size, dropout_rate)
        # Keep the adapter module present in every variant so identity, static, and
        # adaptive runs share an identical parameter surface and initialization.
        self.temporal_adapter = AdaptiveTemporalScaleAdapter(
            input_dim, scales=adapter_scales, mode=adapter_mode
        )
        
        self.action_head = nn.Conv1d(input_dim, num_classes, 1)

    def action_seg_loss(self, samples):
        ce_loss = 0
        for i, sample in enumerate(samples):
            logits = sample["action_logits"]
            labels = sample["framewise_labels"]
            mask = labels != -1
            ce_loss += self.action_ce(logits.permute(1, 0)[mask], labels[mask].to(self.device))
            # smoothing loss
            ce_loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(logits.permute(1, 0)[1:, :], dim=1), F.log_softmax(logits.detach().permute(1, 0)[:-1, :], dim=1)), min=0, max=16))
            add_mask = labels == -1
            if add_mask.sum() > 0:
                ce_loss += torch.sum(- torch.softmax(logits.permute(1, 0)[add_mask], dim=1) * torch.log(torch.softmax(logits.permute(1, 0)[add_mask], dim=1)), dim=1).mean()
        return ce_loss / len(samples)
    
    def gtg2vid_loss(self, samples):
        action_ce_loss = self.action_seg_loss(samples)
        return {
            "action_ce_loss": action_ce_loss
        }


    def forward(self, x, labels=None):
        x = x.permute(0, 2, 1)
        x_ = self.conv_in(x)
        features = self.module(x_, x)
        features = self.temporal_adapter(features)
        action_logits = self.action_head(features)
        return action_logits, features


class AdaptiveTemporalScaleAdapter(nn.Module):
    """Offline per-frame routing over pooled temporal copies of a [B,C,T] feature."""

    VALID_MODES = {"identity", "adaptive", "static"}

    def __init__(self, channels, scales=(1, 2, 4, 8), mode="identity"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"adapter mode must be one of {sorted(self.VALID_MODES)}")
        if tuple(scales) != (1, 2, 4, 8):
            raise ValueError("the locked ATS contract uses scales (1, 2, 4, 8)")
        self.channels = int(channels)
        self.scales = tuple(int(scale) for scale in scales)
        self.mode = mode
        self.gate = nn.Conv1d(self.channels * len(self.scales), len(self.scales), 1)
        self.depthwise = nn.Conv1d(
            self.channels, self.channels, 3, padding=1, groups=self.channels
        )
        self.pointwise = nn.Conv1d(self.channels, self.channels, 1)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.last_gate = None
        # Detached tensors are retained only for training diagnostics.
        self.last_input = None
        self.last_residual = None

    def _branches(self, x):
        length = x.shape[-1]
        branches = []
        for scale in self.scales:
            if scale == 1:
                branch = x
            else:
                kernel = min(scale, length)
                pooled = F.avg_pool1d(
                    x,
                    kernel_size=kernel,
                    stride=kernel,
                    ceil_mode=True,
                    count_include_pad=False,
                )
                branch = F.interpolate(
                    pooled, size=length, mode="linear", align_corners=False
                )
            branches.append(branch)
        return branches

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected [B,{self.channels},T], received {tuple(x.shape)}"
            )
        if self.mode == "identity":
            self.last_gate = None
            self.last_input = None
            self.last_residual = None
            return x

        branches = self._branches(x)
        logits = self.gate(torch.cat(branches, dim=1))
        if self.mode == "static":
            logits = logits.mean(dim=-1, keepdim=True).expand_as(logits)
        weights = torch.softmax(logits, dim=1)
        fused = torch.zeros_like(x)
        for index, branch in enumerate(branches):
            fused = fused + weights[:, index : index + 1] * branch
        residual = self.pointwise(self.depthwise(fused))
        self.last_gate = weights.detach()
        self.last_input = x.detach()
        self.last_residual = residual.detach()
        return x + self.alpha * residual

class MixedConvAttModuleV2(nn.Module): # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i

        self.final_layer = nn.Conv1d(num_f_maps, input_dim_cross, 1, 1, 0)
    
    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:,:,None]

        for layer in self.layers:
            x = layer(x, x_cross)

        # return x
        return self.final_layer(x)

class MixedConvAttentionLayerV2(nn.Module):
    
    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()
        
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x, x_cross):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x, x_cross):
        
        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)
                
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out
