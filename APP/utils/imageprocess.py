import base64
import io
from PIL import Image

MAX_IMAGE_SIZE_MB = 1  # 目标最大尺寸 (MB)
MAX_IMAGE_SIZE = MAX_IMAGE_SIZE_MB * 1024 * 1024  # 转换为字节

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def compress_image(image_file, target_size=MAX_IMAGE_SIZE):
    """ 压缩图片至目标大小 """
    image = Image.open(image_file)
    
    # 转换为 RGB（防止某些 PNG 透明通道问题）
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    
    quality = 90  # 初始质量
    while True:
        img_io = io.BytesIO()
        image.save(img_io, format="JPEG", quality=quality)
        img_size = img_io.tell()
        
        if img_size <= target_size or quality <= 10:
            break  # 达到目标大小或质量过低
        
        quality -= 10  # 降低质量继续压缩

    img_io.seek(0)
    return img_io