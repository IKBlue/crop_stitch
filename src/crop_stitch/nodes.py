import torch

class CropStitch:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "output_target_width": ("INT", {
                    "default": 512, "min": 64, "max": 1024, "step": 64
                }),
                "output_target_height": ("INT", {
                    "default": 512, "min": 64, "max": 1024, "step": 64
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STITCHER")
    RETURN_NAMES = ("Cropped_image", "Stitcher")
    FUNCTION = "crop"
    CATEGORY = "custom/Marco"

    def crop(self, image, mask, output_target_width, output_target_height):
        B, H, W, C = image.shape

        coords = torch.nonzero(mask[0] > 0)
        if coords.shape[0] == 0:
            cx, cy = W // 2, H // 2
        else:
            y_min = torch.min(coords[:, 0]).item()
            x_min = torch.min(coords[:, 1]).item()
            y_max = torch.max(coords[:, 0]).item()
            x_max = torch.max(coords[:, 1]).item()
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

        crop_w = output_target_width
        crop_h = output_target_height

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        if x2 > W:
            x2 = W
            x1 = max(0, x2 - crop_w)
        if y2 > H:
            y2 = H
            y1 = max(0, y2 - crop_h)

        cropped = image[:, y1:y2, x1:x2, :].clone()

        stitcher = {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "canvas_image": image.clone()
        }

        return (cropped, stitcher)


class StitchDirect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cropped_image": ("IMAGE",),
                "stitcher": ("STITCHER",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "custom/Marco"

    def stitch(self, cropped_image, stitcher):
        x, y, w, h = stitcher["x"], stitcher["y"], stitcher["w"], stitcher["h"]
        canvas = stitcher["canvas_image"].clone()
        canvas[:, y:y+h, x:x+w, :] = cropped_image
        return (canvas,)


NODE_CLASS_MAPPINGS = {
    "CropAndStitchFixed": CropStitch,
    "StitchDirect": StitchDirect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAndStitchFixed": "Marco Crop ",
    "StitchDirect": "Marco Stitch",
}
