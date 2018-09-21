from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import cv2


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(sample):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    (pic, target) = sample
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return (img.float().div(255), target)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return (torch.from_numpy(nppic), target)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return (img.float().div(255), target)
    else:
        return (img, target)


def to_pil_image(sample, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    (pic, target) = sample
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    print(npimg.shape)

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return (Image.fromarray(npimg, mode=mode), target)


def normalize(sample, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    (tensor, target) = sample
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return (tensor, target)

def resize_target(sample, size, interpolation=Image.BILINEAR):
    """ resize the target to size, img remains the same
    Args:
        sample (tuple)
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        sample
    """
    (img, target) = sample
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise typeerror('got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return sample
        if w < h:
            ow = size
            oh = int(size * h / w)
            return (img, cv2.resize(target, (ow, oh)))
        else:
            oh = size
            ow = int(size * w / h)
            return (img, cv2.resize(target, (ow, oh)))
    else:
        return (img, cv2.resize(target, size[::-1]))

   

def resize(sample, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    (img, target) = sample
    if not _is_pil_image(img):
        raise typeerror('img should be pil image. got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise typeerror('got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return sample
        if w < h:
            ow = size
            oh = int(size * h / w)
            return (img.resize((ow, oh), interpolation), cv2.resize(target, (ow, oh)))
        else:
            oh = size
            ow = int(size * w / h)
            return (img.resize((ow, oh), interpolation), cv2.resize(target, (ow, oh)))
    else:
        return (img.resize(size[::-1], interpolation), cv2.resize(target, size[::-1]))


def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)


def crop(sample, i, j, h, w):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return (img.crop((j, i, j + w, i + h)), target[i:i+h, j:j+w, ...])


def center_crop(sample, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))

    (img, target) = sample
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(sample, i, j, th, tw)


def resized_crop(sample, i, j, h, w, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.

    Notably used in RandomResizedCrop.

    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    (img, target) = sample
    assert _is_pil_image(img), 'img should be PIL Image'
    sample = crop(sample, i, j, h, w)
    sample = resize(sample, size, interpolation)
    
    return sample


def hflip(sample):
    """Horizontally flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Horizontall flipped image.
    """
    swap = [0,1,2,3,4,5,10,11,12,9,6,7,8,13,14]
    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    target = cv2.flip(target, 1)[...,swap]

    return (img, target)


def vflip(sample):
    """Vertically flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return (img.transpose(Image.FLIP_TOP_BOTTOM), cv2.flip(target, 0))

def adjust_brightness(sample, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return (img, target)


def adjust_contrast(sample, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return (img, target)


def adjust_saturation(sample, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return (img, target)


def adjust_hue(sample, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    (img, target) = sample
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return (img, sample)


def adjust_gamma(sample, gamma, gain=1):
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255) ** gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return (img, target)


def rotate(sample, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.


    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): In degrees degrees counter clockwise order.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    (img, target) = sample
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    #TODO better rotation
    (rows, cols, _) = target.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    target = cv2.warpAffine(target,M,(cols,rows))
    return (img.rotate(angle, resample, expand, center), target)

