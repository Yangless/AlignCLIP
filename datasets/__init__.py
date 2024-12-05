from .hake import build as build_hake
from .mscoco import MSCOCODataset
from .hicodet import HICODETDataset
from .vcoco import VCOCODataset
from .flickr30k import build as build_flickr30k
def build_dataset(split, args):
    return build_hake(split, args)


def build_dataset(split, args):
    if args.dataset == 'mscoco':
        return build_mscoco(split, args)
    elif args.dataset == 'hico_det':
        return build_hico_det(split, args)
    elif args.dataset == 'flickr30k':
        return build_flickr30k(split, args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
