import timm
import torch
from torch import nn
from .metablock import MetaBlock


class TIMMModel(nn.Module):
    def __init__(self, model_name, num_class, neurons_reducer_block=256, p_dropout=0.5,
                comb_method=None, comb_config=None, n_feat_conv=576, initial_weights_path=None):
    
        super().__init__()
        
        n_feat_conv_fused = n_feat_conv * 2
        
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=not initial_weights_path,
            num_classes=0,
        )

        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if isinstance(comb_config, int):
                    self.comb_feat_maps = 32
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

        self.classifier = nn.Linear(n_feat_conv_fused + _n_meta_data, num_class)
        self.reducer_block = None

    def forward(self, img_clinical, img_dermatoscope, meta_data=None, return_features=False):

        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        # Extração (TIMM já retorna achatado quando num_classes=0)
        feat_clin = self.feature_extractor(img_clinical)
        feat_derm = self.feature_extractor(img_dermatoscope)

        feat_clin = feat_clin.view(feat_clin.size(0), -1)
        feat_derm = feat_derm.view(feat_derm.size(0), -1)

        # Fusão
        x = torch.cat((feat_clin, feat_derm), dim=1)

        if self.comb == None:
            pass # Sem reducer block implementado nativamente nesta classe original
        elif isinstance(self.comb, MetaBlock):
            x = x.view(x.size(0), self.comb_feat_maps, -1)  
            x = self.comb(x, meta_data.float())  
            x = x.view(x.size(0), -1) 
            
        if return_features:
            return x
            
        return self.classifier(x)