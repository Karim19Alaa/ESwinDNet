"""
Implementation of ESwinDNet for image demoireing
"""


from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from utils.weight_init import trunc_normal_
from archs.swin_utils import calculate_mask, calculate_rpi_sa, RSTB, PatchEmbed, PatchUnEmbed


class ESwinDNet(nn.Module):
    def __init__(self,
                 img_size,
                 en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 depth,
                 num_heads,
                 window_size,
                 rstb_count=1,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 patch_size=1,
                 resi_connection='1conv',
                 sam_number=1,
                 shared=True
                 ):
        super(ESwinDNet, self).__init__()
        swin_params = {
                 'rstb_count': rstb_count,
                 'img_size': img_size,
                 'embed_dim': en_feature_num,
                 'depth': depth,
                 'num_heads': num_heads,
                 'window_size': window_size,
                 'mlp_ratio': mlp_ratio,
                 'qkv_bias': qkv_bias,
                 'qk_scale': qk_scale,
                 'drop': drop,
                 'norm_layer': norm_layer,
                 'use_checkpoint': use_checkpoint,
                 'patch_size': patch_size,
                 'resi_connection': resi_connection,
                 'shared': shared,
        }
        self.window_size = window_size
        relative_position_index_SA = calculate_rpi_sa(window_size=window_size)
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num,
                                sam_number=sam_number, swin_params=swin_params)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number, swin_params=swin_params)
      
        
        self.apply(self._init_weights)
        

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        attn_mask = dict()
        for i in range(3, 6):
            dims = tuple(dim // 2**i for dim in x_size)
            attn_mask[f'{dims}'] = calculate_mask(dims, self.window_size).to(x.device)

        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
        y_1, y_2, y_3 = self.encoder(x, params)

        out_1, out_2, out_3 = self.decoder(y_1, y_2, y_3, params)

        return out_1, out_2, out_3
    

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.normal_(0.0, 0.02)
                
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, 0.02)
            
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number, swin_params):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, 3, sam_number, swin_params=swin_params)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, 2, sam_number, swin_params=swin_params)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, 1, sam_number, swin_params=swin_params)

    def forward(self, y_1, y_2, y_3, params):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3, params)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2, params)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, params, feat=False)

        return out_1, out_2, out_3


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number, swin_params):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number, swin_params=swin_params)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number, swin_params=swin_params)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number, swin_params=swin_params)

    def forward(self, x, params):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x, params)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1, params)
        out_feature_3 = self.encoder_3(down_feature_2, params)

        return out_feature_1, out_feature_2, out_feature_3



def getSAM(level, in_channel, swin_params, d_list, inter_num):
    if level == 3:
        return SwinSAM(in_channel=in_channel, swin_params=swin_params)
    
    return SAM(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number, swin_params):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        swin_params_ = swin_params.copy()
        swin_params_['img_size'] = swin_params['img_size'] // 2 ** level
        for _ in range(sam_number):
            sam_block = getSAM(level, in_channel=feature_num, swin_params=swin_params_, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x, params):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature, params)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number, swin_params):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        swin_params_ = swin_params.copy()
        swin_params_['img_size'] = swin_params['img_size'] // 2 ** level
        for _ in range(sam_number):
            sam_block = getSAM(level, in_channel=feature_num, swin_params=swin_params_, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, params, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x, params)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SwinSAM(nn.Module):
    def __init__(self, in_channel, swin_params):
        super(SwinSAM, self).__init__()
        swin_params_ = swin_params.copy()
        swin_params_['embed_dim'] = in_channel
        shared = swin_params_.pop('shared')
        if shared:
            self.basic_blocks = nn.ModuleList([AttentionBlock(**swin_params_)] * 3)
        else:
            self.basic_blocks = nn.ModuleList()
            for _ in range(3):
                self.basic_blocks.append(AttentionBlock(**swin_params_))
                swin_params_['img_size'] = swin_params_['img_size'] // 2

        self.fusion = CSAF(3 * in_channel)

    def forward(self, x, params):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_blocks[0](x_0, params)
        y_2 = self.basic_blocks[1](x_2, params)
        y_4 = self.basic_blocks[2](x_4, params)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y
    
class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x, params):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class AttentionBlock(nn.Module):
    def __init__(self, 
                 img_size,
                 embed_dim,
                 depth,
                 num_heads,
                 window_size,
                 rstb_count=1,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 patch_size=1,
                 resi_connection='1conv',
                 ) -> None:
        super().__init__()

        self.mlp_ratio = mlp_ratio
        self.embed_dim = embed_dim

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer)

        self.rstbs = nn.ModuleList([
            RSTB(
                    dim=self.embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,  # no impact on SR results
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection)

            for _ in range(rstb_count)
        ])
        # self.rstb = RSTB(
        #             dim=self.embed_dim,
        #             input_resolution=(patches_resolution[0], patches_resolution[1]),
        #             depth=depth,
        #             num_heads=num_heads,
        #             window_size=window_size,
        #             mlp_ratio=self.mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,  # no impact on SR results
        #             norm_layer=norm_layer,
        #             downsample=None,
        #             use_checkpoint=use_checkpoint,
        #             img_size=img_size,
        #             patch_size=patch_size,
        #             resi_connection=resi_connection)
        

        self.norm = norm_layer(self.embed_dim)
        self.pos_drop = nn.Dropout(p=drop)



    def forward(self, x, params):
        input = x
        x_size = (x.shape[2], x.shape[3])
        params_ = {'attn_mask': params['attn_mask'][f'{x_size}'], 'rpi_sa': params['rpi_sa']}
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for rstb in self.rstbs:
            x = rstb(x, x_size, params_)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        if len(self.rstbs) > 1:
            x = x + input

        return x


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out
    
