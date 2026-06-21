# -*- coding: utf-8 -*-

import torch
from torch import nn
from .metablock import MetaBlock


class MyResnet (nn.Module):

    def __init__(self, resnet, num_class, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=2048):

        super(MyResnet, self).__init__()
        
        # O vetor fundido terá o dobro do tamanho (2048 clínica + 2048 dermato = 4096)
        n_feat_conv_fused = n_feat_conv * 2 

        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if isinstance(comb_config, int):
                    self.comb_feat_maps = 64
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

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer (Agora recebe n_feat_conv_fused, ou seja, 4096)
        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv_fused, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.reducer_block = None

        # Classifier
        if neurons_reducer_block > 0:
            self.classifier = nn.Linear(neurons_reducer_block + _n_meta_data, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv_fused + _n_meta_data, num_class)

    def freeze_base(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def unfreeze_deep_layers(self):
        for param in self.features.parameters():
            param.requires_grad = False
        for name, param in self.features.named_parameters():
            if "layer4" in name: 
                param.requires_grad = True

    # NOVA ASSINATURA: Recebe as duas imagens e a flag return_features
    def forward(self, img_clinical, img_dermatoscope, meta_data=None, return_features=False):

        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        # 1. Extrai características das duas modalidades
        feat_clin = self.features(img_clinical)
        feat_derm = self.features(img_dermatoscope)

        # 2. Achatamento (Flatten)
        feat_clin = feat_clin.view(feat_clin.size(0), -1)
        feat_derm = feat_derm.view(feat_derm.size(0), -1)

        # 3. Fusão (Concatenação das duas modalidades na dimensão das features)
        x = torch.cat((feat_clin, feat_derm), dim=1)

        # 4. Aplicação do MetaBlock e Redução
        if self.comb == None:
            if self.reducer_block is not None:
                x = self.reducer_block(x)  
        elif isinstance(self.comb, MetaBlock):
            # Usando view com -1 para evitar quebras por erro de dimensionalidade
            x = x.view(x.size(0), self.comb_feat_maps, -1) 
            x = self.comb(x, meta_data.float()) 
            x = x.view(x.size(0), -1) 
            if self.reducer_block is not None:
                x = self.reducer_block(x) 

        # 5. Retorna as características unificadas PARA O SVM (Artigo)
        if return_features:
            return x

        # 6. Retorna predição normal da rede (Pré-treinamento da CNN)
        return self.classifier(x)