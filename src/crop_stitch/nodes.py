import torch

class CropStitch:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
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
                "draw_red_box": ("BOOLEAN", {"default": False}),
                "box_color_r": ("INT", {"default": 255, "min": 0, "max": 255}),
                "box_color_g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "box_color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
                "pad_top": ("INT", {"default": 5, "min": 0, "max": 100}),
                "pad_bottom": ("INT", {"default": 5, "min": 0, "max": 100}),
                "pad_left": ("INT", {"default": 5, "min": 0, "max": 100}),
                "pad_right": ("INT", {"default": 5, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STITCHER")
    RETURN_NAMES = ("Cropped_image", "Stitcher")
    FUNCTION = "crop"
    CATEGORY = "custom/Marco"

    def crop(self, image, mask, output_target_width, output_target_height,
             draw_red_box, box_color_r, box_color_g, box_color_b,
             line_thickness, pad_top, pad_bottom, pad_left, pad_right):

        B, H, W, C = image.shape
        coords = torch.nonzero(mask[0] > 0)

        if coords.shape[0] == 0:
            x_min, x_max = W // 2 - 50, W // 2 + 50
            y_min, y_max = H // 2 - 50, H // 2 + 50
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

        # Crop first
        cropped = image[:, y1:y2, x1:x2, :].clone()

        def safe_draw(img, y1, y2, x1, x2, color):
            y1 = max(0, min(y1, img.shape[1]-1))
            y2 = max(0, min(y2, img.shape[1]))
            x1 = max(0, min(x1, img.shape[2]-1))
            x2 = max(0, min(x2, img.shape[2]))
            if y2 > y1 and x2 > x1:
                img[:, y1:y2, x1:x2, :] = color

        if draw_red_box:
            box_color = torch.tensor([
                box_color_r / 255.0,
                box_color_g / 255.0,
                box_color_b / 255.0
            ], device=image.device).view(1, 1, 1, 3)

            # 计算红框位置在 cropped 中的相对坐标
            box_x1 = max(0, x_min - pad_left - x1)
            box_y1 = max(0, y_min - pad_top - y1)
            box_x2 = min(crop_w - 1, x_max + pad_right - x1)
            box_y2 = min(crop_h - 1, y_max + pad_bottom - y1)
            t = line_thickness

            # 绘制红框在 cropped 上
            safe_draw(cropped, box_y1, box_y1 + t, box_x1, box_x2 + 1, box_color)  # top
            safe_draw(cropped, box_y2 - t + 1, box_y2 + 1, box_x1, box_x2 + 1, box_color)  # bottom
            safe_draw(cropped, box_y1, box_y2 + 1, box_x1, box_x1 + t, box_color)  # left
            safe_draw(cropped, box_y1, box_y2 + 1, box_x2 - t + 1, box_x2 + 1, box_color)  # right

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Cropped_image": ("IMAGE",),
                "Stitcher": ("STITCHER",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "custom/Marco"

    def stitch(self, Cropped_image, Stitcher):
        x, y, w, h = Stitcher["x"], Stitcher["y"], Stitcher["w"], Stitcher["h"]
        canvas = Stitcher["canvas_image"].clone()
        canvas[:, y:y+h, x:x+w, :] = Cropped_image
        return (canvas,)


NODE_CLASS_MAPPINGS = {
    "CropAndStitchFixed": CropStitch,
    "StitchDirect": StitchDirect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAndStitchFixed": "Marco Crop",
    "StitchDirect": "Marco Stitch",
}
