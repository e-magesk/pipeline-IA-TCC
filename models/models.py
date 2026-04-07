from .effnet import MyEffnet
from .mobilenet import MyMobilenet
from .resnet import MyResnet
from .vggnet import MyVGGNet
from torchvision import models
from efficientnet_pytorch import EfficientNet
from .timmmodel import TIMMModel

###############################################################################################
# Variables and Constants
###############################################################################################
_MODELS_CONFIG = {
    'resnet-50': 'ResNet50_Weights.DEFAULT',
    'resnet-101': 'ResNet101_Weights.DEFAULT',
    'densenet-121': 'DenseNet121_Weights.DEFAULT',
    'densenet-169': 'DenseNet169_Weights.DEFAULT',
    'vgg-13': 'VGG13_BN_Weights.DEFAULT',
    'vgg-16': 'VGG16_Weights.DEFAULT',
    'vgg-19': 'VGG19_BN_Weights.DEFAULT',
    'mobilenet': 'MobileNet_V2_Weights.DEFAULT',    
    'efficientnet-b4': True,
    'efficientnet-b3': True,
    'coat_small': {
        'weights': 'coat_small.in1k', 
        'n_feat_conv': 320,  
    },
    'pit_s_distilled_224': {
        'weights': 'pit_s_distilled_224.in1k',
        'n_feat_conv': 576,
    },
    'nextvit_small': {
        'weights': 'nextvit_small.bd_ssld_6m_in1k',
        'n_feat_conv': 1024,
    },
    'volo_d1_224': {
        'weights': 'volo_d1_224.sail_in1k',
        'n_feat_conv': 384,
    },
    'hgnet_small': {
        'weights': 'hgnet_small.ssld_in1k',
        'n_feat_conv': 2048,
    },
    'maxvit_tiny': {
        'weights': 'maxvit_tiny_tf_224.in1k',
        'n_feat_conv': 512,
    },
    'gcvit_tiny': {
        'weights': 'gcvit_tiny.in1k',
        'n_feat_conv': 512,
    },
    'davit_tiny': {
        'weights': 'davit_tiny.msft_in1k',
        'n_feat_conv': 768,
    },
    'caformer_s18': {
        'weights': 'caformer_s18.sail_in1k',
        'n_feat_conv': 512,
    },
    'efficientformerv2_l': {
        'weights': 'efficientformerv2_l.snap_dist_in1k',
        'n_feat_conv': 384,
    },
    'mvitv2_small': {
        'weights': 'mvitv2_small.fb_in1k',
        'n_feat_conv': 768,
    },
    'xcit_small_12_p8_224': {'weights': 'xcit_small_12_p8_224.fb_dist_in1k',
        'n_feat_conv': 384,
    },
    'pvt_v2_b2_li': {
        'weights': 'pvt_v2_b2_li.in1k',
        'n_feat_conv': 512,
    },
    'coat_lite_small': {
        'weights': 'coat_lite_small.in1k',
        'n_feat_conv': 512,
    },
    'swinv2_cr_tiny_ns_224': {
        'weights': 'swinv2_cr_tiny_ns_224.sw_in1k',
        'n_feat_conv': 768,
    },
    'swin_s3_tiny_224': {
        'weights': 'swin_s3_tiny_224.ms_in1k',
        'n_feat_conv': 768,
    },
    'convformer_s18': {
        'weights': 'convformer_s18.sail_in1k',
        'n_feat_conv': 512,
    },
    'caformer_b36': {
        'weights': 'caformer_b36.sail_in1k',
        'n_feat_conv': 768,
    },
    'caformer_s36': {
        'weights': 'caformer_s36.sail_in1k',
        'n_feat_conv': 512,
    },
    'caformer_m36': {
        'weights': 'caformer_m36.sail_in1k',
        'n_feat_conv': 576,
    },
    'eva02_base_patch14_448':  {
        'weights': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
        'n_feat_conv': 768,
    },
    'hgnetv2_b6': {
        'weights': 'hgnetv2_b6.ssld_stage2_ft_in1k',
        'n_feat_conv': 2048,
    },
    'nextvit_large': {
        'weights': 'nextvit_large.bd_ssld_6m_in1k',
        'n_feat_conv': 1024,
    },
    'pit_b_distilled_224': {
        'weights': 'pit_b_distilled_224.in1k',
        'n_feat_conv': 1024,
    },
    'mobilenet-v4': {
        'weights': 'mobilenetv4_hybrid_medium.e500_r224_in1k',
        'n_feat_conv': 1280,
    }
}

CONFIG_METABLOCK_BY_MODEL = {
    'caformer_s18': 16, 
    'caformer_m36': 18,
    'caformer_s36': 16,
    'caformer_b36': 24,
    'pit_b_distilled_224': 32,
    'pit_s_distilled_224': 18, 
    'efficientformerv2_l': 12,
    'coat_small': 10,
    'coat_lite_small': 16,
    'davit_tiny': 24,
    'gcvit_tiny': 16,
    'hgnetv2_b6': 64,
    'hgnet_small': 64,
    'nextvit_small': 32,
    'nextvit_large': 32,
    'mobilenet-v4': 40,
    'tiny-vit': 18,
    'convformer_s18': 16,
    'mvitv2_small': 24, 
    'xcit_small_12_p8_224': 12,
    'pvt_v2_b2_li': 16,
    'swinv2_cr_tiny_ns_224': 24,
    'swin_s3_tiny_224': 24,
    'maxvit_tiny': 16,
    'volo_d1_224': 12,
    'resnet-50': 64,
    'mobilenet': 40,
    'efficientnet-b4': 56,
    'efficientnet-b0': 40,
    'efficientnet-b5': 64,
}

_NORM_AND_SIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [224, 224]]
###############################################################################################


def set_class_model (model_name, num_class, neurons_reducer_block=0, comb_method=None, 
                     comb_config=None, pretrained=True, freeze_conv=False, initial_weights_path = None):
    """
    This function returns the model object based on the model name and other parameters. It works for 
    classification tasks.

    :param model_name: str
        The name of the model to be used. Refer to _MODELS for available models.
    :param num_class: int
        The number of classes in the dataset.
    :param neurons_reducer_block: int
        The number of neurons in the reducer block. Default is 0.
    :param comb_method: str
        The method to combine the features. Default is None.
    :param comb_config: dict
        The configuration of the comb method. Default is None.
    :param pretrained: bool
        Whether to use the pretrained weights from ImageNet. Default is True.
    :param freeze_conv: bool
        Whether to freeze the convolutional layers. Default is False.
    :param initial_weights_path: str | Path
        Path of the initial weights of the model. Default is None.
    :return: object
        The model object.    
    """

    if pretrained:
        pre_torch = _MODELS_CONFIG[model_name]
    else:
        pre_torch = None

    if model_name not in _MODELS_CONFIG.keys():
        raise Exception(f"The model {model_name} is not available!")

    model = None
    if model_name == 'resnet-50':
        model = MyResnet(models.resnet50(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'resnet-101':
        model = MyResnet(models.resnet101(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'densenet-121':
        model = MyDensenet(models.densenet121(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'densenet-169':
        model = MyDensenet(models.densenet169(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config, n_feat_conv=1664)
    elif model_name == 'vgg-13':
        model = MyVGGNet(models.vgg13_bn(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'vgg-16':
        model = MyVGGNet(models.vgg16_bn(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'vgg-19':
        model = MyVGGNet(models.vgg19_bn(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'mobilenet':
        model = MyMobilenet(models.mobilenet_v2(weights=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name in ['efficientnet-b4', 'efficientnet-b3']:
        if pretrained:
            model = MyEffnet(EfficientNet.from_pretrained(model_name), num_class, neurons_reducer_block, freeze_conv,
                             comb_method=comb_method, comb_config=comb_config)
        else:
            model = MyEffnet(EfficientNet.from_name(model_name), num_class, neurons_reducer_block, freeze_conv,
                             comb_method=comb_method, comb_config=comb_config)
    elif _MODELS_CONFIG[model_name]:
        model = TIMMModel(_MODELS_CONFIG[model_name]['weights'], num_class, neurons_reducer_block, freeze_conv, 
                          comb_method=comb_method, comb_config=comb_config,
                          n_feat_conv = _MODELS_CONFIG[model_name]['n_feat_conv'], initial_weights_path=initial_weights_path)
                             
    return model

