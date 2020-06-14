
import albumentations as A
from albumentations.pytorch import ToTensor
from pprint import pprint

flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

# can we use IAA series here?

pipeset = [
    [ # uint8 only
        # channe
        A.InvertImg(always_apply=False, p=0.5),
        A.Posterize(num_bits=4, always_apply=False, p=0.5),
        # normalize, CLAHE = histogram equalize
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        A.Equalize(mode='cv', by_channels=True, mask=None,
            mask_params=(), always_apply=False, p=0.5),
        # noise
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5),
            always_apply=False, p=0.5),
    ],

    [ # blur
        A.Blur(blur_limit=7, always_apply=False, p=0.5),
        A.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
        A.MedianBlur(blur_limit=5, always_apply=False, p=0.5),
        A.GaussianBlur(blur_limit=7, always_apply=False, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, always_apply=False, p=0.5),
    ],

    [ # nature
        A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5),
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5),
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.5),
    ],

    [ # normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ],

    [ # 飽和度/色彩相關
        # will normalize by max or mean brightness
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
        A.RandomContrast(limit=0.2, always_apply=False, p=0.5),
        A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5),
    ],

    [ # affine
        A.VerticalFlip(always_apply=False, p=0.5),
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.Flip(always_apply=False, p=0.5),
        A.RandomRotate90(always_apply=False, p=0.5),
        A.Rotate(limit=180, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        A.Transpose(always_apply=False, p=0.5),
    ],

    [ # 形變
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, p=0.5),
    ],

    [ # grid dropout or shuffle
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, always_apply=False, p=0.5),
        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
        A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=1.0),
    ],

    [ # noise
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
    ], 

    [ # channel
        A.ChannelShuffle(always_apply=False, p=0.5),
        A.ToGray(always_apply=False, p=0.5),
        A.Solarize(threshold=128, always_apply=False, p=0.5),
    ],

    #[ # compression
    #    A.JpegCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
    #    A.ImageCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5),
    #    # compression_type=A.ImageCompressionType.JPEG, 
    #],

    [ # scale and crop
        A.Compose(
            [A.CenterCrop(height=128, width=128, always_apply=False, p=1.0),
            A.Resize(height=256, width=256, interpolation=1, always_apply=False, p=1)], p = 1),

        # hand adjust here
        A.RandomResizedCrop(height=256, width=256, scale=(0.9, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=1.0),
    ],

    #[ # bbox
    #    A.RandomSizedBBoxSafeCrop(height=256, width=256, erosion_rate=0.0, interpolation=1, always_apply=False, p=1.0),
    #],
 
]

flattend_pipeset = flatten(pipeset) 

def numOfPipeline():
    return len(pipeset) + sum([len(p) for p in pipeset]) * 2

pipeline_length = 16
# pipeline_length = numOfPipeline() # full pipeline
nature_beg = 2
normal_beg = 3
nature_index = sum([len(x) for x in pipeset[:nature_beg]])
normalize_index = sum([len(x) for x in pipeset[:normal_beg]])
bc_start = sum([len(x) for x in pipeset[:normal_beg + 1]])

def idxList2trainPipeline(index_list):

    index_list = index_list[:pipeline_length]

    post_add = []
    g_post_add = []
    pipeline = []
    single_pipes = sum([len(p) for p in pipeset]) * 2
    group_pipes = len(pipeset)

    for idx in index_list:
        if idx >= single_pipes:
            group_idx = idx - single_pipes
            if group_idx == normal_beg:
                g_post_add.append(group_idx)
            elif group_idx < normal_beg:
                pipeline.insert(0, A.OneOf(pipeset[group_idx], p = 0.3))
            else:
                pipeline.append(A.OneOf(pipeset[group_idx], p = 0.3))
        elif idx % 2 == 0:
            single_idx = idx // 2
            if single_idx > normalize_index:
                pipeline.append(flattend_pipeset[idx // 2])
            elif single_idx == normalize_index:
                post_add.append(normalize_index)
            else:
                pipeline.insert(0, flattend_pipeset[idx // 2])

    for idx in post_add:
        pipeline.append(flattend_pipeset[idx])

    for idx in g_post_add:
        pipeline.append(A.OneOf(pipeset[idx], p = 0.3))

    pipeline.insert(0, A.Resize(height=256, width=256, interpolation=1, always_apply=False, p=1))
    pipeline.append(A.Resize(height=128, width=128, interpolation=1, always_apply=False, p=1))
    pipeline.append(ToTensor())
    return A.Compose(pipeline, p = 1)

def idxList2validPipeline(index_list):
    
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

def printPipeline(idxList, idx2pipe):
    print("pipeline: ", idxList[:pipeline_length])
    pprint(A.to_dict(idx2pipe(idxList)))
