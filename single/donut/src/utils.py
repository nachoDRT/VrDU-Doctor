import cv2
import numpy as np


def add_transparent_image(
    background, foreground, alpha_factor=1.0, x_offset=None, y_offset=None
):
    """
    Function sourced from StackOverflow contributor Ben.

    This function was found on StackOverflow and is the work of Ben, a contributor
    to the community. We are thankful for Ben's assistance by providing this useful
    method.

    Original Source:
    https://stackoverflow.com/questions/40895785/
    using-opencv-to-overlay-transparent-image-onto-another-image
    """

    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert (
        bg_channels == 3
    ), f"background image should have exactly 3 channels (RGB). found:{bg_channels}"
    assert (
        fg_channels == 4
    ), f"foreground image should have exactly 4 channels (RGBA). found:{fg_channels}"

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y : fg_y + h, fg_x : fg_x + w]
    background_subsection = background[bg_y : bg_y + h, bg_x : bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    foreground_colors = cv2.cvtColor(foreground_colors, cv2.COLOR_BGR2RGB)
    alpha_channel = foreground[:, :, 3] / 255 * alpha_factor  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = (
        background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    )

    # overwrite the section of the background image that has been updated
    background[bg_y : bg_y + h, bg_x : bg_x + w] = composite

    return background


def convert_tensor_to_rgba_image(tensor):

    saliency_array = tensor.cpu().numpy()

    # Normalize img a 0-255
    if saliency_array.dtype != np.uint8:
        saliency_array = (255 * saliency_array / saliency_array.max()).astype(np.uint8)

    heatmap = cv2.applyColorMap(saliency_array, cv2.COLORMAP_JET)

    # Pixels are transparent where no saliency [128, 0, 0] is black in COLORMAP_JET
    alpha_channel = np.ones(heatmap.shape[:2], dtype=heatmap.dtype) * 255
    black_pixels_mask = np.all(heatmap == [128, 0, 0], axis=-1)
    alpha_channel[black_pixels_mask] = 0

    # Combinar los canales RGB y alfa
    saliency_rgba = cv2.merge((heatmap, alpha_channel))

    return saliency_rgba


def convert_rgb_to_rgba_image(image):

    alpha_channel = np.ones(image.shape[:2], dtype=image.dtype) * 255
    rbga = cv2.merge((cv2.cvtColor(image, cv2.COLOR_RGB2BGR), alpha_channel))

    return rbga
