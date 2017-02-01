from __future__ import division

import numpy as np
import scipy.ndimage as ndi
from skimage import color


def criminisi(img, mask, patch_size=(9, 9),
              multichannel=False, verbose=False):
    """Inpaint masked points in image using Criminisi et al. algorithm.
    This function performs constrained synthesis using Criminisi et al. [1]_.
    It grows the texture of the known regions to reconstruct unknown pixels.
    Parameters
    ----------
    img : (M, N[, C]) ndarray
        Input image.
    mask : (M, N) ndarray
        Array of pixels to be inpainted. Have to be the same shape as one
        of the 'img' channels. Unknown pixels have to be represented with 1,
        known pixels - with 0.
    patch_size : 2-tuple of uint, optional
        Size of the neighborhood window. Patch centered at the pixel to be
        inpainted will be used. Refer to Notes section for the details on
        value choice. Has to be positive and odd.
    multichannel : boolean, optional
        If True, the last `img` dimension is considered as a color channel,
        otherwise as spatial.
    verbose : boolean, optional
        If True, prints the number of pixels left to be inpainted.
    Returns
    -------
    out : (M, N[, C]) ndarray
        Input image with masked pixels inpainted.
    Notes
    -----
    For best results, ``patch_size`` should be larger in size than the largest
    texel (texture element) being inpainted. A texel is the smallest repeating
    block of pixels in a texture or pattern.
    For general purpose usage the default value is recommended.
    References
    ----------
    .. [1] A. Criminisi, P. Perez, and K. Toyama. 2004. Region filling and
           object removal by exemplar-based image inpainting. Trans. Img. Proc.
           13, 9 (September 2004), 1200-1212. DOI=10.1109/TIP.2004.833105.
    Example
    -------
    >>> from skimage.data import checkerboard
    >>> img = checkerboard()
    >>> mask = np.zeros_like(image, dtype=np.bool)
    >>> mask[75:125, 75:125] = 1
    >>> img[mask] = 0
    >>> out = inpaint_criminisi(img, mask)

    """

    img_baseshape = img.shape[:-1] if multichannel else img.shape

    if len(img_baseshape) != 2:
        raise ValueError('Only single- or multi-channel 2D images are supported.')

    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(img):
        raise TypeError('Masked arrays are not supported')

    if not all([dim % 2 for dim in patch_size]):
        raise ValueError("All values in `patch_size` have to be odd.")

    # img = skimage.img_as_float(img)
    mask = mask.astype(np.uint8)

    if multichannel:
        # Images in CIE Lab colour space are more perceptually uniform
        out = color.rgb2lab(img)
    else:
        out = img[..., np.newaxis]

    patch_area = patch_size[0] * patch_size[1]
    patch_arm_col = int((patch_size[0] - 1) / 2)
    patch_arm_row = int((patch_size[1] - 1) / 2)

    # Pad image and mask to ease edge pixels processing
    out = np.pad(out, ((patch_arm_col, patch_arm_col),
                       (patch_arm_row, patch_arm_row),
                       (0, 0)),
                 mode='constant')
    tmp = np.pad(mask, ((patch_arm_col, patch_arm_col),
                        (patch_arm_row, patch_arm_row)),
                  mode='constant', constant_values=2)

    mask = (tmp != 0)
    source_region = (tmp == 0).astype(np.float)
    target_region = (tmp == 1).astype(np.bool)

    # Assign np.nan to unknown pixels to ease gradient computation
    out_nan = out.astype(np.float)
    out_nan[mask, :] = np.nan

    # Calculate data_term normalization constant
    alpha = np.nanmax(out_nan) - np.nanmin(out_nan)
    if alpha == 0:
        alpha = 1

    # Create an array of potential sample centers
    source_region_valid = ndi.filters.minimum_filter(
        source_region, footprint=np.ones(patch_size), mode='constant')

    # Create a grid of patch relative coordinates
    patch_grid_row, patch_grid_col = \
        np.mgrid[-patch_arm_col:patch_arm_col + 1,
                 -patch_arm_row:patch_arm_row + 1]

    # Perform initialization
    fill_front = np.bitwise_xor(target_region,
                                ndi.morphology.binary_dilation(target_region))
    if verbose:
        step = 0

    while np.any(fill_front):
        if step == 0:
            confidence_terms = source_region.astype(np.float)

            grad_y_ch, grad_x_ch = np.gradient(out_nan, axis=(0, 1))
            grad_y = np.sum(grad_y_ch, axis=2)
            grad_x = np.sum(grad_x_ch, axis=2)

            norm_y, norm_x = np.gradient(source_region)
        else:
            # Update the working matrices
            fill_front = np.bitwise_xor(
                target_region, ndi.morphology.binary_dilation(target_region))

            grad_y_ch, grad_x_ch = np.gradient(out_nan, axis=(0, 1))
            grad_y = np.sum(grad_y_ch, axis=2)
            grad_x = np.sum(grad_x_ch, axis=2)

            norm_y, norm_x = np.gradient(np.bitwise_not(target_region))

        # Rotate gradient by 90deg
        grad_x, grad_y = grad_y, -grad_x
        grad_mod = grad_x ** 2 + grad_y ** 2

        # Perform gradient nanmax-pooling
        grad_x_pool = np.zeros_like(grad_x)
        grad_y_pool = np.zeros_like(grad_y)

        for idx_r in range(patch_size[0], out.shape[0] - patch_size[0]):
            for idx_c in range(patch_size[1], out.shape[1] - patch_size[1]):
                grad_mod_roi = grad_mod[idx_r + patch_grid_row,
                                        idx_c + patch_grid_col]

                if np.all(np.isnan(grad_mod_roi)):
                    grad_x_pool[idx_r, idx_c] = np.nan
                    grad_y_pool[idx_r, idx_c] = np.nan
                else:
                    idx_max = np.nanargmax(grad_mod_roi)
                    idx_max_r, idx_max_c = \
                        np.unravel_index(idx_max, grad_mod_roi.shape)

                    grad_x_pool[idx_r, idx_c] = \
                        grad_x[idx_r + idx_max_r - patch_arm_col,
                               idx_c + idx_max_c - patch_arm_row]
                    grad_y_pool[idx_r, idx_c] = \
                        grad_y[idx_r + idx_max_r - patch_arm_col,
                               idx_c + idx_max_c - patch_arm_row]

        # Calculate data_terms
        data_terms = np.abs(norm_x * grad_x_pool + norm_y * grad_y_pool) / alpha

        # Calculate priorities and pick the top-priority patch
        priorities = confidence_terms * data_terms * fill_front
        prio_r, prio_c = np.unravel_index(np.nanargmax(priorities),
                                          priorities.shape)

        # Find the exemplar with the minimal distance
        distances = np.zeros_like(source_region_valid) + 1e16

        for tmp_r, tmp_c in zip(*np.where(source_region_valid)):
            distances[tmp_r, tmp_c] = np.nansum(np.abs(
                out_nan[prio_r + patch_grid_row,
                        prio_c + patch_grid_col, :] ** 2 - \
                out_nan[tmp_r + patch_grid_row,
                        tmp_c + patch_grid_col, :] ** 2))

        best_r, best_c = np.unravel_index(np.nanargmin(distances),
                                          distances.shape)

        # Copy image data
        to_update = target_region[prio_r + patch_grid_row,
                                  prio_c + patch_grid_col]
        out_nan[prio_r + patch_grid_row * to_update,
                prio_c + patch_grid_col * to_update] = \
            out[best_r + patch_grid_row * to_update,
                best_c + patch_grid_col * to_update]
        out[prio_r + patch_grid_row * to_update,
            prio_c + patch_grid_col * to_update] = \
            out[best_r + patch_grid_row * to_update,
                best_c + patch_grid_col * to_update]

        # Update confidence_terms
        confidence_terms[prio_r + patch_grid_row * to_update,
                         prio_c + patch_grid_col * to_update] = \
            np.nansum(confidence_terms[prio_r + patch_grid_row,
                                       prio_c + patch_grid_col]) / patch_area

        # Update mask
        target_region[prio_r + patch_grid_row,
                      prio_c + patch_grid_col] = False

        if verbose:
            if step % 10 == 0:
                print('Pixels left/total: {}/{}'.format(
                    np.sum(target_region), target_region.size))
            step += 1

    out = out[patch_arm_col:-patch_arm_col+1,
              patch_arm_row:-patch_arm_row+1, :]

    if multichannel:
        out = color.lab2rgb(out)
    else:
        out = out[..., 0]

    return out
