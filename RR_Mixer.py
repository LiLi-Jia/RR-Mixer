import torch
import os
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from transformers import RobertaModel, RobertaConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
def get_mask_from_sequence(sequence, dim):
    return torch.sum(torch.abs(sequence), dim=dim) == 0

def to_cpu(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, norm=nn.LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x):

        return self.fn(self.norm(x)) + x
class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out,drop=0.1):
        super(FeedForward, self).__init__()
        self.re1 = Rearrange('b c h w -> b h w c')
        self.fc1 = nn.Linear(dim_in,hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(drop)
        self.fc2 =nn.Linear(hidden_dim,dim_out, bias=False)
        self.re2 =Rearrange('b h w c -> b c h w')
    def forward(self, x):
        x1 = self.re1(x)
        x2 = self.fc1(x1)
        x3 = self.activation(x2)
        x4 = self.dropout(x3)
        x5 = self.fc2(x4)
        x6 = self.dropout(x5)
        x7 = self.re2(x6)
        return x7+x
class CrossRegion(nn.Module):
    def __init__(self, step=1, dim=1):
        super().__init__()
        self.step = step
        self.dim = dim

    def forward(self, x):
        return torch.roll(x, self.step, self.dim)


class InnerRegionW(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        self.region = nn.Sequential(
            Rearrange('b c h (w group) -> b (c w) h group', w=self.w)
        )

    def forward(self, x):
        return self.region(x)


class InnerRegionH(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.region = nn.Sequential(
            Rearrange('b c (h group) w -> b (c h) group w', h=self.h)
        )

    def forward(self, x):
        return self.region(x)


class InnerRegionRestoreW(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        self.region = nn.Sequential(
            Rearrange('b (c w) h group -> b c h (w group)', w=self.w)
        )

    def forward(self, x):
        return self.region(x)


class InnerRegionRestoreH(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.region = nn.Sequential(
            Rearrange('b (c h) group w -> b c (h group) w', h=self.h)
        )

    def forward(self, x):
        return self.region(x)

class Mixer_Layer(nn.Module):
    def __init__(self, h, w, d_model, cross_region_step=1, cross_region_id=0, cross_region_interval=2,
                 padding_type='circular'):
        super().__init__()

        assert (padding_type in ['constant', 'reflect', 'replicate', 'circular'])
        self.padding_type = padding_type
        self.w = w
        self.h = h

        # cross region every cross_region_interval Mixer_Layer
        self.cross_region = (cross_region_id % cross_region_interval == 0)

        if self.cross_region:
            self.cross_regionW = CrossRegion(step=cross_region_step, dim=3)
            self.cross_regionH = CrossRegion(step=cross_region_step, dim=2)
            self.cross_region_restoreW = CrossRegion(step=-cross_region_step, dim=3)
            self.cross_region_restoreH = CrossRegion(step=-cross_region_step, dim=2)
        else:
            self.cross_regionW = nn.Identity()
            self.cross_regionH = nn.Identity()
            self.cross_region_restoreW = nn.Identity()
            self.cross_region_restoreH = nn.Identity()

        self.inner_regionW = InnerRegionW(w)
        self.inner_regionH = InnerRegionH(h)
        self.inner_region_restoreW = InnerRegionRestoreW(w)
        self.inner_region_restoreH = InnerRegionRestoreH(h)

        self.proj_h = FeedForward(h * d_model, d_model // 2, h * d_model)
        self.proj_w = FeedForward(w * d_model, d_model // 2, w * d_model)
        self.proj_c = FeedForward(d_model, d_model, d_model)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        B, C, H, W = x.shape
        padding_num_w = W % self.w
        padding_num_h = H % self.h
        x = nn.functional.pad(x, (0, self.w - padding_num_w, 0, self.h - padding_num_h), self.padding_type)

        x_h = self.inner_regionH(self.cross_regionH(x))
        x_w = self.inner_regionW(self.cross_regionW(x))

        x_h = self.proj_h(x_h)
        x_w = self.proj_w(x_w)
        x_c = self.proj_c(x)
###恢复原有维度
        x_h = self.cross_region_restoreH(self.inner_region_restoreH(x_h))
        x_w = self.cross_region_restoreW(self.inner_region_restoreW(x_w))
###三者进行拼接
        out = x_h + x_w +x_c
###恢复padding之前得维度
        out = out[:, :, 0:H, 0:W]
        out = out.permute(0, 2, 3, 1)
        return out

class Mixer_Block(nn.Module):
    def __init__(self, h, w, d_model_in, d_model_out, depth, cross_region_step, cross_region_interval,
                 expansion_factor=2, dropout=0.1, pooling=False, padding_type='circular'):
        super().__init__()
        self.fc1 = nn.Linear(d_model_in,d_model_out)
        self.pooling = pooling
        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model_in, nn.Sequential(
                    Mixer_Layer(
                        h, w, d_model_in, cross_region_step=cross_region_step, cross_region_id=i_depth + 1,
                        cross_region_interval=cross_region_interval, padding_type=padding_type
                    )
                ), norm=nn.LayerNorm),
                PreNormResidual(d_model_in, nn.Sequential(
                    nn.Linear(d_model_in, d_model_in * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model_in * expansion_factor, d_model_in),
                    nn.Dropout(dropout),
                ), norm=nn.LayerNorm),
            ) for i_depth in range(depth)]
        )

    def forward(self, x):
        x = self.model(x)
        if self.pooling:
            x = self.fc1(x)
        return x

class RR_Mixer_Module(nn.Module):
    def __init__(
            self, opt,
            num_classes=3,
            d_model=[256,256,256,256],
            h=[4, 3, 3, 2],
            w=[4, 3, 3, 2],
            cross_region_step=[2, 2, 1, 1],
            cross_region_interval=2,
            depth=[1,1,2,1],
            expansion_factor=2,
            padding_type='circular',
    ):
        super().__init__()
        d_t, d_a, d_v, d_common, encoders, top_k= opt.d_t, opt.d_a, opt.d_v, opt.d_common, opt.encoders, opt.top_k
        self.d_t, self.d_a, self.d_v, self.d_common, self.encoders, self.top_k = d_t, d_a, d_v, d_common, encoders, top_k
        self.layers = nn.ModuleList()
        for i_layer in range(len(depth)):
            i_depth = depth[i_layer]
            i_stage = Mixer_Block(h[i_layer], w[i_layer], d_model[i_layer],
                                   d_model_out=d_model[i_layer + 1] if (i_layer + 1 < len(depth)) else d_model[-1],
                                   depth=i_depth, cross_region_step=cross_region_step[i_layer],
                                   cross_region_interval=cross_region_interval,
                                   expansion_factor=expansion_factor, pooling=((i_layer + 1) < len(depth)),
                                   padding_type=padding_type)
            self.layers.append(i_stage)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model[-1]),
            Reduce('b h w c -> b c', 'mean'),
            nn.Linear(d_model[-1], num_classes)
        )
         # Projector
        self.W_t197 = nn.Linear(d_v, opt.d_common, bias=False)

        # LayerNormalize & Dropout
        self.ln_a, self.ln_v = nn.LayerNorm(opt.d_common, eps=1e-6), nn.LayerNorm(opt.d_common, eps=1e-6)
        self.dropout_t, self.dropout_a, self.dropout_v = nn.Dropout(opt.dropout[0]), nn.Dropout(
            opt.dropout[1]), nn.Dropout(opt.dropout[2])


        assert self.encoders in ['lstm', 'gru', 'conv']
        # Extractors
        if self.encoders == 'conv':
            self.conv_a = nn.Conv1d(in_channels=d_a, out_channels=d_common, kernel_size=3, stride=1, padding=1)
            self.conv_v = nn.Conv1d(in_channels=d_v, out_channels=d_common, kernel_size=3, stride=1, padding=1)
        elif self.encoders == 'lstm':
            self.rnn_t = nn.LSTM(d_t, d_common, 1, bidirectional=True, batch_first=True)
            self.rnn_a = nn.LSTM(d_a,d_common, 1, bidirectional=True, batch_first=True)
        elif self.encoders == 'gru':
            self.rnn_t = nn.GRU(d_t, d_common, 2, bidirectional=True, batch_first=True)
            self.rnn_a = nn.GRU(d_a, d_common, 2, bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        robertaconfig = RobertaConfig.from_pretrained("roberta-large",output_hidden_states=True)
        self.robertamodel = RobertaModel.from_pretrained('roberta-large',config=robertaconfig)
    def forward(self, bert_sentence_details_list,bert_target_details_list,v,debug=False):
        a = self.robertamodel(input_ids=bert_target_details_list[0],attention_mask=bert_target_details_list[1])[0]
        l_av = a.shape[1]
        t = self.robertamodel(input_ids=bert_sentence_details_list[0],attention_mask=bert_sentence_details_list[1])[0]
        if debug:
            print('Origin:', t.shape, a.shape, v.shape)
        mask_t = bert_sentence_details_list[1]
        # Compute cosine similarity between text and image features
        list=[]
        for j in range(v.size(1)):
            similarities = F.cosine_similarity(t.unsqueeze(2),v[:, j, :].unsqueeze(1).unsqueeze(1),dim=-1)
            list.append(similarities)
        # Combine the weighted similarities to obtain the final image representation
        combined_similarities = torch.stack(list, dim=2).squeeze(-1)

        # Get the top-k most similar images for each text
        _, top_k_indices = torch.topk(combined_similarities, self.top_k, dim=2)
        image_features_selected = torch.gather(v.unsqueeze(1).expand(-1,self.top_k, -1, -1),2,top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_v))
        top_v = image_features_selected.view(image_features_selected.shape[0], -1, self.d_v)
        top_v = top_v.view(top_v.shape[0], -1, self.d_v)
        top_v = top_v.permute(0, 2, 1)
        le1 = nn.Linear(top_v.shape[2], mask_t.shape[1], bias=False).to(device=0)
        top_v = le1(top_v)
        v = top_v.permute(0, 2, 1)
        v = self.W_t197(v)   #经过筛选之后的图像特征表示

        # Pad audio & video
        length_padded = t.shape[1]
        pad_before = int((length_padded - l_av) / 2)
        pad_after = length_padded - l_av - pad_before
        a = F.pad(a, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)
        a_fill_pos = (get_mask_from_sequence(a, dim=-1).int() * mask_t).bool()
        a = a.masked_fill(a_fill_pos.unsqueeze(-1), 1e-6)
        if debug:
            print('Padded:', t.shape, a.shape, v.shape)
        lengths = to_cpu(bert_sentence_details_list[1]).sum(dim=1)
        l_av_padded = a.shape[1]

        # Extract features
        if self.encoders == 'conv':
            a, t = self.conv_a(a.transpose(1, 2)).transpose(1, 2), self.conv_v(t.transpose(1, 2)).transpose(1, 2)
            a, t = F.relu(self.ln_a(a)), F.relu(self.ln_v(t))
        elif self.encoders in ['lstm', 'gru']:

            a = pack_padded_sequence(a, lengths, batch_first=True, enforce_sorted=False)
            t = pack_padded_sequence(t, lengths, batch_first=True, enforce_sorted=False)
            self.rnn_a.flatten_parameters()
            self.rnn_t.flatten_parameters()
            (packed_a, a_out), (packed_t, t_out) = self.rnn_a(a),self.rnn_t(t)
            a, _ = pad_packed_sequence(packed_a, batch_first=True, total_length=l_av_padded)
            t, _ = pad_packed_sequence(packed_t, batch_first=True, total_length=l_av_padded)
            if False:
                print('After RNN', a.shape, v.shape)
            if self.encoders == 'lstm':
                a_out, t_out = a_out[0], t_out[0]
            a = torch.stack(torch.split(a, self.d_common, dim=-1), -1).sum(-1)
            t = torch.stack(torch.split(t, self.d_common, dim=-1), -1).sum(-1)
            if False:
                print('After Union', a.shape, v.shape)
            a, t = F.relu(self.ln_a(a)), F.relu(self.ln_v(t))
        else:
            raise NotImplementedError

        t, a, v = self.dropout_t(t), self.dropout_a(a), self.dropout_v(v)

        if False:
            print('After Extracted', t.shape, a.shape, v.shape)
        x = torch.stack([t, a, v], dim=1)

        embedding = x
        for layer in self.layers:
            embedding = layer(embedding)
        out = self.mlp_head(embedding)
        return out