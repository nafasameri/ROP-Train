from PIL import Image, ImageOps
import numpy as np
import cv2
from mask import *

def adjust_brightness(img, alpha=1.0, beta=50):
    """
    Adjust brightness and contrast of an image.
    alpha: Contrast control (1.0-3.0)
    beta: Brightness control (0-100)
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def gauss_blur_recursive(img, sigma):
    pass

def get_gauss_kernel(sigma, dim=2):
    ksize = int(np.floor(sigma * 6) / 2) * 2 + 1
    k_1D = np.arange(ksize) - ksize // 2
    k_1D = np.exp(-k_1D**2 / (2 * sigma**2))
    k_1D = k_1D / np.sum(k_1D)
    if dim == 1:
        return k_1D
    elif dim == 2:
        return k_1D[:, None].dot(k_1D.reshape(1, -1))

def gauss_blur_original(img, sigma):
    row_filter = get_gauss_kernel(sigma, 1)
    t = cv2.filter2D(img, -1, row_filter[..., None])
    return cv2.filter2D(t, -1, row_filter.reshape(1, -1))

def gauss_blur(img, sigma, method='original'):
    if method == 'original':
        return gauss_blur_original(img, sigma)
    elif method == 'recursive':
        return gauss_blur_recursive(img, sigma)

def MultiScaleRetinex(img, sigmas=[150], weights=None, flag=True):
    if weights == None:
        weights = np.ones(len(sigmas)) / len(sigmas)
    elif not abs(sum(weights) - 1) < 0.00001:
        raise ValueError('Sum of weights must be 1!')
    r = np.zeros(img.shape, dtype='double')
    img = img.astype('double')
    for i, sigma in enumerate(sigmas):
        r += (np.log(img + 1) - np.log(gauss_blur(img, sigma) + 1)) * weights[i]
    if flag:
        mmin = np.min(r, axis=(0, 1), keepdims=True)
        mmax = np.max(r, axis=(0, 1), keepdims=True)
        r = (r - mmin) / (mmax - mmin) * 255
        r = r.astype('uint8')
    return r

def retinex_AMSR(img, sigmas=[12, 80, 250]):
    img = img.astype('double') + 1
    msr = MultiScaleRetinex(img - 1, sigmas, flag=False)
    y = 0.05
    for i in range(msr.shape[-1]):
        v, c = np.unique((msr[..., i] * 100).astype('int'), return_counts=True)
        sort_v_index = np.argsort(v)
        sort_v, sort_c = v[sort_v_index], c[sort_v_index]
        zero_ind = np.where(sort_v == 0)[0][0]
        zero_c = sort_c[zero_ind]
        _ = np.where(sort_c[:zero_ind] <= zero_c * y)[0]
        if len(_) == 0:
            low_ind = 0
        else:
            low_ind = _[-1]
        _ = np.where(sort_c[zero_ind + 1:] <= zero_c * y)[0]
        if len(_) == 0:
            up_ind = len(sort_c) - 1
        else:
            up_ind = _[0] + zero_ind + 1
        low_v, up_v = sort_v[[low_ind, up_ind]] / 100
        msr[..., i] = np.maximum(np.minimum(msr[:, :, i], up_v), low_v)
        mmin = np.min(msr[..., i])
        mmax = np.max(msr[..., i])
        msr[..., i] = (msr[..., i] - mmin) / (mmax - mmin) * 255
    msr = msr.astype('uint8')
    return msr


def perproccessing(img_path, output_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)

    # Apply retinex_AMSR function to the image
    processed_img = retinex_AMSR(img)

    # Adjust the brightness of the processed image
    processed_img = adjust_brightness(processed_img, alpha=1.2, beta=50)

    processed_img = apply_circular_mask(processed_img)

    # Save the processed image to the output folder
    cv2.imwrite(output_path, processed_img)
