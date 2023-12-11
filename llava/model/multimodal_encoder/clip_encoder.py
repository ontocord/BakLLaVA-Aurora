import torch
import torch.nn as nn

from multilingual_clip import pt_multilingual_clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
from transformers import ClapModel, ClapProcessor
import transformers


#This is the audio tower
class ClapConfig(transformers.PretrainedConfig):
    model_type = "Clap"

    def __init__(self, modelBase='laion/clap-htsat-unfused', transformerDimSize=1024, imageDimSize=768, **kwargs):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        self.modelBase = modelBase
        super().__init__(**kwargs)

class ClapTower(transformers.PreTrainedModel):
    config_class = ClapConfig

    def __init__(self, modelBase, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.clap =  ClapModel.from_pretrained(modelBase) # 
        self.LinearTransformation = torch.nn.Linear(in_features=config.transformerDimensions,
                                                    out_features=config.numDims)

    def forward(self, *args, **vargs):
        embs = self.clap(*args, **vargs)
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []
        

class MultilingualCLIP(pt_multilingual_clip.MultilingualCLIP):
        
    def load_model(self, model_name):
        self.textemb_tower = self.forward
        self.textemb_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name , cache_dir="/p/scratch/ccstdl/raj3")
        self.transformer.requires_grad_(False)
        self.LinearTransformation.requires_grad_(False)
    
    @torch.no_grad()
    def forward(self, text_tok):
        embs = self.transformer(**text_tok)[0]
        att = text_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return nn.functional.normalize(self.LinearTransformation(embs))


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name, cache_dir="/p/scratch/ccstdl/raj3")
        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, cache_dir="/p/scratch/ccstdl/raj3")
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return nn.functional.normalize(image_features)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
