import torchvision.transforms as T
from torchvision.transforms import functional as F

class ResizePad224:
    def __call__(self, img):
        w, h = img.size              # PIL: (width, height)
        scale = 224 / max(w, h)      # dopasuj dłuższy bok
        new_w, new_h = int(w*scale), int(h*scale)
        img = F.resize(img, (new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)

        pad_left  = (224 - new_w) // 2
        pad_top   = (224 - new_h) // 2
        pad_right = 224 - new_w - pad_left
        pad_bot   = 224 - new_h - pad_top
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bot), fill=0)
        return img
