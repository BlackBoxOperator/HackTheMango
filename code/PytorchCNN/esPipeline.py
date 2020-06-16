
import albumentations as A
from albumentations.pytorch import ToTensor
from pprint import pprint

fixed = lambda x: x
paried = lambda x: (x, x)
sorted_tuple = lambda x: tuple(sorted(x))
flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
OneOfP = 0.3

# can we use IAA series here?

pipeline_length = 12
# pipeline_length = numOfPipeline() # full pipeline
nature_beg = 2
normal_beg = 3

def CenterCrop(height=128, width=128, p=1.0):
    return A.Compose(
            [A.CenterCrop(height=height, width=width, always_apply=False, p=p),
            A.Resize(height=256, width=256, interpolation=1, always_apply=False, p=1)], p = 1)

CenterCrop.get_class_fullname = lambda: 'albumentations.augmentations.transforms.CenterCrop'

pipes = [
    [ # uint8 only
        # channel
        A.InvertImg,
        A.Posterize,
        # normalize, CLAHE = histogram equalize
        A.CLAHE,
        A.Equalize,
        # noise
        A.ISONoise,
    ],

    [ # blur
        A.Blur,
        A.MotionBlur,
        A.MedianBlur,
        A.GaussianBlur,
        A.RandomGamma,
    ],

    [ # nature nature_beg = 2
        A.RandomSnow,
        A.RandomRain,
        A.RandomFog,
        A.RandomSunFlare,
        A.RandomShadow,
    ],

    [ # normalize normal_beg = 3
        A.Normalize,
    ],

    [ # 飽和度/色彩相關
        # will normalize by max or mean brightness
        A.RandomBrightnessContrast,
        A.HueSaturationValue,
        A.RGBShift,
        A.RandomBrightness,
        A.RandomContrast,
        A.ChannelDropout,
    ],

    [ # affine
        A.VerticalFlip,
        A.HorizontalFlip,
        A.Flip,
        A.RandomRotate90,
        A.Rotate,
        A.ShiftScaleRotate,
        A.Transpose,
    ],

    [ # 形變
        A.OpticalDistortion,
        A.GridDistortion,
        A.ElasticTransform,
    ],

    [ # grid dropout or shuffle
        A.CoarseDropout,
        A.Cutout,
        A.RandomGridShuffle,
    ],

    [ # noise
        A.GaussNoise,
    ], 

    [ # channel
        A.ChannelShuffle,
        A.ToGray,
        A.Solarize,
    ],

    #[ # compression
    #    A.JpegCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
    #    A.ImageCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
    #    # compression_type=A.ImageCompressionType.JPEG, 
    #],

    [ # scale and crop
        CenterCrop,

        # hand adjust here
        A.RandomResizedCrop,
    ],

    #[ # bbox
    #    A.RandomSizedBBoxSafeCrop(height=256, width=256, erosion_rate=0.0, interpolation=1, always_apply=False, p=1.0),
    #],
]

pipes_name = [ {p.get_class_fullname(): 0 for p in group } for group in pipes ]
# pprint(pipeattr)

pipes_attr =  {
    'albumentations.augmentations.transforms.CLAHE': {
      'p': [float, (0.0, 1.0), 0.5,  fixed],
      'clip_limit': [float, (0.005, 8.0), 4.0, fixed],
      'tile_grid_size': [int, (3, 25), 8, paried],
    },
    'albumentations.augmentations.transforms.Equalize': {
      'p': [float, (0.0, 1.0), 0.5,  fixed],
    },
    'albumentations.augmentations.transforms.ISONoise': {
      'p': [float, (0.0, 1.0), 0.5,  fixed],
      'color_shift': [(float, float), ((0.0, 1.0), (0.0, 1.0)), (0.01, 0.05), sorted_tuple],
      'intensity': [(float, float), ((0.0, 1.0), (0.0, 1.0)), (0.1, 0.5), fixed],
    },
    'albumentations.augmentations.transforms.InvertImg': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.Posterize': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'num_bits': [int, (0, 8), 4, fixed],
    },
    'albumentations.augmentations.transforms.Blur': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'blur_limit': [int, (3, 100), 7, fixed],
    },
    'albumentations.augmentations.transforms.GaussianBlur': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'blur_limit': [int, (3, 100), 7, fixed],
    },
    'albumentations.augmentations.transforms.MedianBlur': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'blur_limit': [int, (3, 100), 5, fixed],
    },
    'albumentations.augmentations.transforms.MotionBlur': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'blur_limit': [int, (3, 100), 7, fixed],
    },
    'albumentations.augmentations.transforms.RandomGamma': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'gamma_limit': [(int, int), ((-130, 130), (-130, 130)), (80, 120), sorted_tuple],
    },
    'albumentations.augmentations.transforms.RandomFog': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.RandomRain': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.RandomShadow': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.RandomSnow': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.RandomSunFlare': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.Normalize': {
      'p': [float, (0.0, 1.0), 1.0, fixed],
      'mean': [(float, float, float),
               ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0)), 
               (0.485, 0.456, 0.406), fixed],
      'std': [(float, float, float),
               ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0)), 
               (0.229, 0.224, 0.225), fixed],
    },
    'albumentations.augmentations.transforms.ChannelDropout': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.HueSaturationValue': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'hue_shift_limit': [int, (5, 50), 20, fixed],
      'sat_shift_limit': [int, (5, 50), 30, fixed],
      'val_shift_limit': [int, (5, 50), 20, fixed],
    },
    'albumentations.augmentations.transforms.RGBShift': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'r_shift_limit': [int, (5, 50), 20, fixed], 
      'g_shift_limit': [int, (5, 50), 20, fixed],
      'b_shift_limit': [int, (5, 50), 20, fixed],
    },
    'albumentations.augmentations.transforms.RandomBrightness': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'limit': [float, (0.0, 1.0), 0.2, fixed],
    },
    'albumentations.augmentations.transforms.RandomBrightnessContrast': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'brightness_limit': [float, (0.0, 1.0), 0.2, fixed],
      'contrast_limit': [float, (0.0, 1.0), 0.2, fixed],
      'brightness_by_max': [int, (0, 1), 1, bool],
    },
    'albumentations.augmentations.transforms.RandomContrast': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'limit': [float, (0.0, 1.0), 0.2, fixed],
    },
    'albumentations.augmentations.transforms.Flip': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.HorizontalFlip': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.RandomRotate90': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.Rotate': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'limit': [int, (0, 180), 90, fixed],
      'interpolation': [int, (0, 4), 1, fixed],
    },
    'albumentations.augmentations.transforms.ShiftScaleRotate': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'shift_limit': [float, (0.0, 1.0), 0.0625, fixed],
      'scale_limit': [float, (0.0, 1.0), 0.1, fixed],
      'rotate_limit': [int, (0, 180), 45, fixed],
      'interpolation': [int, (0, 4), 1, fixed],
    },
    'albumentations.augmentations.transforms.Transpose': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.VerticalFlip': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.ElasticTransform': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'alpha': [float, (0, 5), 1, fixed],
      'sigma': [float, (25, 75), 50, fixed],
      'alpha_affine': [float, (25, 75), 50, fixed],
      'interpolation': [int, (0, 4), 1, fixed],
    },
    'albumentations.augmentations.transforms.GridDistortion': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'num_steps': [int, (3, 9), 5, fixed],
      'distort_limit': [float, (0, 1), 0.3, fixed],
      'interpolation': [int, (0, 4), 1, fixed],
    },
    'albumentations.augmentations.transforms.OpticalDistortion': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'distort_limit': [float, (0.0, 1.0), 0.05, fixed],
      'shift_limit': [float, (0.0, 1.0), 0.05, fixed],
      'interpolation': [int, (0, 4), 1, fixed],
    },
    'albumentations.augmentations.transforms.CoarseDropout': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'max_holes':  [int, (1, 16), 8, fixed], 
      'max_height': [int, (1, 16), 8, fixed],
      'max_width':  [int, (1, 16), 8, fixed],
    },
    'albumentations.augmentations.transforms.Cutout': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'num_holes':  [int, (1, 16), 8, fixed],
      'max_h_size': [int, (1, 16), 8, fixed],
      'max_w_size': [int, (1, 16), 8, fixed],
    },
    'albumentations.augmentations.transforms.RandomGridShuffle': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'grid': [int, (1, 64), 3, paried],
    },
    'albumentations.augmentations.transforms.GaussNoise': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'var_limit': [(float, float), ((5, 60), (5, 60)), (10.0, 50.0), fixed],
      'mean': [float, (0.0, 1.0), 0.0, fixed],
    },
    'albumentations.augmentations.transforms.ChannelShuffle': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.Solarize': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'threshold': [int, (0, 255), 128, fixed],
    },
    'albumentations.augmentations.transforms.ToGray': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
    },
    'albumentations.augmentations.transforms.RandomResizedCrop': {
      'p': [float, (0.0, 1.0), 0.5, fixed],
      'height': [int, (256, 256), 256, fixed],
      'width':  [int, (256, 256), 256, fixed],
    },
    'albumentations.augmentations.transforms.CenterCrop': {
      'p': [float, (0.0, 1.0), 1, fixed],
      'height': [int, (64, 240), 128, fixed],
      'width': [int, (64, 240), 128, fixed],
    },
}

def get_init_val(attr, dic):
    [typ, rang, init, ret] = dic
    return ret(init)

def get_pipe_attr(p):
    return {
        attrk: get_init_val(attrk, attrv)
        for attrk, attrv in pipes_attr[p.get_class_fullname()].items()
    }

#flattend_pipes_default = flatten(pipes_default) 
flattend_pipes = flatten(pipes) 

nature_index = sum([len(x) for x in pipes[:nature_beg]])
normalize_index = sum([len(x) for x in pipes[:normal_beg]])
bc_start = sum([len(x) for x in pipes[:normal_beg + 1]])

single_pipes = sum([len(p) for p in pipes]) * 2
group_pipes = len(pipes)

def numOfPipeline():
    return len(pipes) + sum([len(p) for p in pipes]) * 2

def idxList2trainPipeline(index_list, reorder = True, cut = True, length = 0):

    if cut:
        if length:
            if isinstance(length, tuple):
                mn, mx = length
                length = sum(index_list[-6:]) % (mx - mn + 1) + mn
            index_list = index_list[:length]
        else:
            index_list = index_list[:pipeline_length]

    post_add = []
    g_post_add = []
    pipeline = []

    for idx in index_list:
        if idx >= single_pipes:
            group_idx = idx - single_pipes
            if group_idx == normal_beg and reorder:
                g_post_add.append(group_idx)
            elif group_idx < normal_beg and reorder:
                pipeline.insert(0, A.OneOf(
                    #pipes_default[group_idx],
                    [p(**get_pipe_attr(p)) for p in pipes[group_idx]],
                    p = OneOfP))
            else:
                pipeline.append(A.OneOf(
                    #pipes_default[group_idx],
                    [p(**get_pipe_attr(p)) for p in pipes[group_idx]],
                    p = OneOfP))
        elif idx % 2 == 0:
            single_idx = idx // 2
            if not reorder:
                p = flattend_pipes[idx // 2]
                pipeline.append(
                    #flattend_pipes_default[idx // 2]
                    p(**get_pipe_attr(p))
                )
            elif single_idx > normalize_index:
                p = flattend_pipes[idx // 2]
                pipeline.append(
                    #flattend_pipes_default[idx // 2]
                    p(**get_pipe_attr(p))
                )
            elif single_idx == normalize_index:
                post_add.append(normalize_index)
            else:
                p = flattend_pipes[idx // 2]
                pipeline.insert(0,
                    #flattend_pipes_default[idx // 2]
                    p(**get_pipe_attr(p))
                )

    for idx in post_add:
        p = flattend_pipes[idx]
        pipeline.append(
            #flattend_pipes_default[idx]
            p(**get_pipe_attr(p))
        )

    for idx in g_post_add:
        pipeline.append(A.OneOf(
            #pipes_default[idx]
            [p(**get_pipe_attr(p)) for p in pipes[idx]],
            p = OneOfP))

    pipeline.insert(0, A.Resize(height=256, width=256, interpolation=1, always_apply=False, p=1))
    pipeline.append(A.Resize(height=128, width=128, interpolation=1, always_apply=False, p=1))
    pipeline.append(ToTensor())
    return A.Compose(pipeline, p = 1)

def idxList2validPipeline(index_list, cut = True, length = 0):
   
    if cut:
        if length:
            if isinstance(length, tuple):
                mn, mx = length
                length = sum(index_list[-6:]) % (mx - mn + 1) + mn
            index_list = index_list[:length]
        else:
            index_list = index_list[:pipeline_length]

    pipeline = [A.Resize(height=128, width=128, interpolation=1, always_apply=False, p=1)]

    if normalize_index in index_list:
        pipeline.append(
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0, always_apply=False, p=1.0))

    if bc_start in index_list:
        pipeline.append(
                A.RandomBrightnessContrast(
                    brightness_limit = 0,
                    contrast_limit = 0,
                    brightness_by_max = False,
                    always_apply = True,
                    p = 1
                ))

    pipeline.append(ToTensor())
    return A.Compose(pipeline, p = 1)

def printPipeline(idxList, index2pipe, length = 0):
    if length:
        if isinstance(length, tuple):
            mn, mx = length
            length = sum(idxList[-6:]) % (mx - mn + 1) + mn
        print("pipeline: ", idxList[:length])
        pprint(A.to_dict(index2pipe(idxList, length = length)))
    else:
        print("pipeline: ", idxList[:pipeline_length])
        pprint(A.to_dict(index2pipe(idxList)))

def idx2pipe(idx, construct = False):
    if idx >= single_pipes:
        group_idx = idx - single_pipes
        return tuple(p for p in pipes[group_idx]) \
                if not construct \
                else A.OneOf([p(**get_pipe_attr(p)) for p in pipes[group_idx]], p = OneOfP)
    elif idx % 2 == 0:
        p = flattend_pipes[idx // 2]
        return p if not construct else p(**get_pipe_attr(p))
    else:
        return None

def toFloatListByRange(typ, rang, df):
    if typ == int or typ == float:
        (mn, mx) = rang
        return [(df - mn) / (mx - mn)] if mx != mn else [1]
    elif type(typ) == tuple:
        return flatten([toFloatListByRange(t, r, d) for t, r, d in zip(typ, rang, df)])
    else:
        print("error"), exit()

def defaultParametersByPipeline(pipeline):
    params = []
    for idx in pipeline:
        p = idx2pipe(idx)
        if type(p) == tuple:
            params.append(OneOfP)
            for subp in p:
                for name, [typ, rang, df, ret] in pipes_attr[subp.get_class_fullname()].items():
                    params.extend(toFloatListByRange(typ, rang, df))
        elif p:
            for name, [typ, rang, df, ret] in pipes_attr[p.get_class_fullname()].items():
                params.extend(toFloatListByRange(typ, rang, df))
    return params

def deNorParams(typ, rang, df, params):
    if typ == int:
        (mn, mx) = rang
        return int(params.pop(0) * (mx - mn) + mn) if mx != mn else mn
    elif typ == float:
        (mn, mx) = rang
        return params.pop(0) * (mx - mn) + mn if mx != mn else mn
    elif type(typ) == tuple:
        return tuple(deNorParams(t, r, d, params) for t, r, d in zip(typ, rang, df))
    else:
        print("error"), exit()

def pakWithResizeTotensor(pipeline):
    pipeline.insert(0, A.Resize(height=256, width=256, interpolation=1, always_apply=False, p=1))
    pipeline.append(A.Resize(height=128, width=128, interpolation=1, always_apply=False, p=1))
    pipeline.append(ToTensor())
    return A.Compose(pipeline, p = 1)

def newPipelineWithParams(pipeline, params):
    params = params.copy()
    cons_pipes = []
    for idx in pipeline:
        p = idx2pipe(idx)
        if type(p) == tuple:
            ofp = params.pop(0)
            ps = []
            for subp in p:
                kw = {}
                for name, [typ, rang, df, ret] in pipes_attr[subp.get_class_fullname()].items():
                    kw[name] = ret(deNorParams(typ, rang, df, params))
                ps.append(subp(**kw))
            cons_pipes.append(A.OneOf(ps, p = ofp))
        elif p:
            kw = {}
            for name, [typ, rang, df, ret] in pipes_attr[p.get_class_fullname()].items():
                kw[name] = ret(deNorParams(typ, rang, df, params))
            cons_pipes.append(p(**kw))

    return pakWithResizeTotensor(cons_pipes)

def printSelectPipes(pipeline):
    ps = [p // 2 for p in pipeline if p < single_pipes]
    gpipe = [p - single_pipes for p in pipeline if p >= single_pipes]
    ps = [flattend_pipes[p].get_class_fullname() for p in ps]
    for g in gpipe:
        ps.append([p.get_class_fullname() for p in pipes[g]])
    pprint(ps)

if __name__ == '__main__':
    #pl = [83, 46, 51, 42, 44, 60, 23, 14, 63, 77, 61, 20, 36, 24, 58, 30]
    #pl = [i for i in range(numOfPipeline())] # ok
    #orgp = idxList2trainPipeline(pl, reorder = False, cut = False)
    #dftp = pakWithResizeTotensor([x for x in [idx2pipe(idx, construct = True) for idx in pl] if x])
    #cvtp = newPipelineWithParams(pl, defaultParametersByPipeline(pl))
    #print(str(orgp), file=open("org", "w"))
    #print(str(dftp), file=open("dft", "w"))
    #print(str(cvtp), file=open("cvt", "w"))
    #print(str(orgp) == str(cvtp))
    printSelectPipes([56, 38, 86, 62])
