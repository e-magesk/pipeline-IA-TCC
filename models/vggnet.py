# -*- coding: utf-8 -*-
import torch
from torch import nn
from .metablock import MetaBlock


class MyVGGNet (nn.Module):

    def __init__(self, vgg, num_class, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=25088):

        super(MyVGGNet, self).__init__()

        n_feat_conv_fused = n_feat_conv * 2

        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if isinstance(comb_config, int):
                    self.comb_feat_maps = 784
                    self.comb = MetaBlock(self.comb_feat_maps, comb_config)
                elif isinstance(comb_config, list):
                    self.comb_feat_maps = comb_config[0]
                    self.comb = MetaBlock(self.comb_feat_maps, comb_config[1])
                else:
                    raise Exception(
                        "comb_config must be a list or int to define the number of feat maps and the metadata")
            else:
                raise Exception("There is no comb_method called " + comb_method + ". Please, check this out.")
        else:
            self.comb = None

        self.features = nn.Sequential(*list(vgg.children())[:-1])

        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv_fused, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),
                nn.Linear(1024, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.reducer_block = None
            
        if neurons_reducer_block > 0:
            self.classifier = nn.Linear(neurons_reducer_block + _n_meta_data, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv_fused + _n_meta_data, num_class)


    def forward(self, img_clinical, img_dermatoscope, meta_data=None, return_features=False):

        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        # Extração
        feat_clin = self.features(img_clinical)
        feat_derm = self.features(img_dermatoscope)

        # Achatamento
        feat_clin = feat_clin.view(feat_clin.size(0), -1)
        feat_derm = feat_derm.view(feat_derm.size(0), -1)

        # Fusão
        x = torch.cat((feat_clin, feat_derm), dim=1)

        if self.comb == None:
            if self.reducer_block is not None:
                x = self.reducer_block(x)  
        elif isinstance(self.comb, MetaBlock):
            x = x.view(x.size(0), self.comb_feat_maps, -1) 
            x = self.comb(x, meta_data.float()) 
            x = x.view(x.size(0), -1) 
            if self.reducer_block is not None:
                x = self.reducer_block(x) 

        if return_features:
            return x

        return self.classifier(x)