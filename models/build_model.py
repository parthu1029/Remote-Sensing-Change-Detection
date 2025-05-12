from .unet_model import build_unet
from .unet_resnet_model import build_unet_with_resnet
from .unet_densenet_model import build_unet_with_densenet

def build_all_models(input_shape, l2_lambda=0.01):
    model_a = build_unet(input_shape, l2_lambda)
    model_b = build_unet_with_resnet(input_shape, l2_lambda)
    model_c = build_unet_with_densenet(input_shape, l2_lambda)
    return model_a, model_b, model_c

def build_unet_model(input_shape, l2_lambda=0.01):
    return build_unet(input_shape, l2_lambda)
    
def build_unet_resnet_model(input_shape, l2_lambda=0.01):
    return build_unet_with_resnet(input_shape, l2_lambda)
    
def build_unet_densenet_model(input_shape, l2_lambda=0.01):
    return build_unet_with_densenet(input_shape, l2_lambda)
