from moco import loader as moco_loader

moco_loader.ensure_rvl_cdip_cached(
    split="train", 
    cache_dir="/lus/eagle/projects/PBML/ziyan/dataset"
)