import os
from .clip_encoder import CLIPVisionTower, MultilingualCLIP


def build_multimodal_towers(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    text_tower = None
    clip_tower = None
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        clip_tower= CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        text_tower_name = getattr(vision_tower_cfg, 'mm_text_tower', getattr(vision_tower_cfg, 'text_tower', None))
        text_tower= MultilingualCLIP.from_pretrained(text_tower_name, cache_dir="/p/scratch/ccstdl/raj3")
        text_tower.load_model(text_tower_name)
    if clip_tower is None:    
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    return clip_tower, text_tower