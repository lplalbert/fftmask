import math
import random

import kornia
import torch
import torch.nn as nn

def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for translation."""
    matrix = translation.new_zeros((translation.shape[0], 3, 3))
    matrix[:, 0, 0] = 1
    matrix[:, 1, 1] = 1
    matrix[:, 2, 2] = 1
    matrix[:, 0, 2] = translation[:, 0]
    matrix[:, 1, 2] = translation[:, 1]
    return matrix

def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor(
        [center_x, center_y],
        device=tensor.device, dtype=tensor.dtype)
    return center

def _compute_scaling_matrix(scale: torch.Tensor,
                            center: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for scaling."""
    angle: torch.Tensor = torch.zeros(scale.shape[0], device=scale.device, dtype=scale.dtype)
    matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
    return matrix

def _compute_rotation_matrix(angle: torch.Tensor,
                             center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones((angle.shape[0], 2), device=angle.device, dtype=angle.dtype)
    matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
    return matrix

def translate(image, device, d=8):
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    
    trans = image.new_empty((c, 2)).uniform_(-d, d)
    
    translation_matrix: torch.Tensor = _compute_translation_matrix(trans)
    matrix = translation_matrix[..., :2, :3]
    
    data_warp = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border')
    
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp

def rotate(image, device, d=8):
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    
    angle = image.new_empty(c).uniform_(-d, d)
    center = image.new_tensor([[w / 2 - 1, h / 2 - 1]]).expand(c, -1)
    
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)
    matrix = rotation_matrix[..., :2, :3]
    
    data_warp = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border')
    
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp

def perspective(image, device, d=8):
    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]
    
    points_src = image.new_tensor([
        [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]
    ]).expand(c, -1, -1)
    
    shifts = image.new_empty((c, 4, 2)).uniform_(-d, d)
    base_dst = image.new_tensor([
        [0., 0.], 
        [w - 1., 0.], 
        [w - 1., h - 1.], 
        [0., h - 1.]
    ]).expand(c, -1, -1)
    
    points_dst = base_dst + shifts
    
    image = image.float()
    M = kornia.get_perspective_transform(points_src.float(), points_dst.float())
    data_warp = kornia.warp_perspective(image, M, dsize=(h, w))
    
    return data_warp

def Light_Distortion(c, embed_image, device):
    device = embed_image.device
    B, C, H, W = embed_image.shape
    a = 0.7 + random.random() * 0.2
    b = 1.1 + random.random() * 0.2
    
    if c == 0:
        direction = random.randint(1, 4)
        i_vec = torch.arange(H, dtype=embed_image.dtype, device=device)
        
        val = -((b - a) / (H - 1)) * (i_vec - W) + a
        mask_2d = val.unsqueeze(1).expand(H, W)
        
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = torch.rot90(mask_2d, 1, [0, 1])
        elif direction == 3:
            O = torch.rot90(mask_2d, 2, [0, 1])
        else:
            O = torch.rot90(mask_2d, 3, [0, 1])
            
        return O.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    else:
        x = random.randint(0, H - 1)
        y = random.randint(0, W - 1)
        
        max_len = math.sqrt(max(
            x**2 + y**2, 
            (x - (H - 1))**2 + y**2, 
            x**2 + (y - (W - 1))**2, 
            (x - (H - 1))**2 + (y - (W - 1))**2
        ))
        
        Y, X = torch.meshgrid(
            torch.arange(H, dtype=embed_image.dtype, device=device),
            torch.arange(W, dtype=embed_image.dtype, device=device),
            indexing='ij'
        )
        
        dist = torch.sqrt((Y - x)**2 + (X - y)**2)
        mask_2d = dist / max_len * (a - b) + b
        
        return mask_2d.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

def Moire_Distortion(embed_image, device):
    device = embed_image.device
    B, C, H, W = embed_image.shape
    
    Y, X = torch.meshgrid(
        torch.arange(1, H + 1, dtype=embed_image.dtype, device=device),
        torch.arange(1, W + 1, dtype=embed_image.dtype, device=device),
        indexing='ij'
    )
    
    channels_to_gen = min(3, C)
    
    theta = embed_image.new_empty((channels_to_gen, 1, 1)).uniform_(0, math.pi)
    center_y = embed_image.new_empty((channels_to_gen, 1, 1)).uniform_(0, H)
    center_x = embed_image.new_empty((channels_to_gen, 1, 1)).uniform_(0, W)
    
    dist = torch.sqrt((Y.unsqueeze(0) - center_y)**2 + (X.unsqueeze(0) - center_x)**2)
    z1 = 0.5 + 0.5 * torch.cos(2 * math.pi * dist)
    
    phase = torch.cos(theta) * X.unsqueeze(0) + torch.sin(theta) * Y.unsqueeze(0)
    z2 = 0.5 + 0.5 * torch.cos(phase)
    
    z = torch.minimum(z1, z2)
    M = (z + 1) / 2
    
    if C > 3:
        padded = embed_image.new_zeros((C, H, W))
        padded[:3] = M
        M = padded
    
    return M.unsqueeze(0).expand(B, -1, -1, -1)

class ScreenShooting(nn.Module):
    def __init__(self):
        super(ScreenShooting, self).__init__()
        self._cache_key = None
        self.register_buffer("_grid_y0", torch.empty(0), persistent=False)
        self.register_buffer("_grid_x0", torch.empty(0), persistent=False)
        self.register_buffer("_grid_y1", torch.empty(0), persistent=False)
        self.register_buffer("_grid_x1", torch.empty(0), persistent=False)
        self.register_buffer("_perspective_src", torch.empty(0), persistent=False)
        self.register_buffer("_perspective_dst_base", torch.empty(0), persistent=False)

    def _prepare_cache(self, image):
        _, _, h, w = image.shape
        key = (image.device, image.dtype, h, w)
        if self._cache_key == key:
            return

        y0 = torch.arange(h, dtype=image.dtype, device=image.device)
        x0 = torch.arange(w, dtype=image.dtype, device=image.device)
        self._grid_y0, self._grid_x0 = torch.meshgrid(y0, x0, indexing='ij')

        y1 = torch.arange(1, h + 1, dtype=image.dtype, device=image.device)
        x1 = torch.arange(1, w + 1, dtype=image.dtype, device=image.device)
        self._grid_y1, self._grid_x1 = torch.meshgrid(y1, x1, indexing='ij')

        self._perspective_src = image.new_tensor([
            [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]
        ]).unsqueeze(0)
        self._perspective_dst_base = self._perspective_src.clone()
        self._cache_key = key

    def _perspective(self, image, d=8):
        b, _, h, w = image.shape
        points_src = self._perspective_src.expand(b, -1, -1)
        shifts = image.new_empty((b, 4, 2)).uniform_(-d, d)
        points_dst = self._perspective_dst_base.expand(b, -1, -1) + shifts

        image = image.float()
        matrix = kornia.get_perspective_transform(points_src.float(), points_dst.float())
        return kornia.warp_perspective(image, matrix, dsize=(h, w))

    def _light_distortion(self, c, image):
        _, _, h, w = image.shape
        a = 0.7 + random.random() * 0.2
        b = 1.1 + random.random() * 0.2

        if c == 0:
            direction = random.randint(1, 4)
            val = -((b - a) / (h - 1)) * (self._grid_y0[:, 0] - w) + a
            mask_2d = val.unsqueeze(1).expand(h, w)

            if direction == 1:
                mask_2d = mask_2d
            elif direction == 2:
                mask_2d = torch.rot90(mask_2d, 1, [0, 1])
            elif direction == 3:
                mask_2d = torch.rot90(mask_2d, 2, [0, 1])
            else:
                mask_2d = torch.rot90(mask_2d, 3, [0, 1])
        else:
            x = random.randint(0, h - 1)
            y = random.randint(0, w - 1)
            max_len = math.sqrt(max(
                x**2 + y**2,
                (x - (h - 1))**2 + y**2,
                x**2 + (y - (w - 1))**2,
                (x - (h - 1))**2 + (y - (w - 1))**2
            ))

            dist = torch.sqrt((self._grid_y0 - x)**2 + (self._grid_x0 - y)**2)
            mask_2d = dist / max_len * (a - b) + b

        return mask_2d.unsqueeze(0).unsqueeze(0)

    def _moire_distortion(self, image):
        _, channels, h, w = image.shape
        channels_to_gen = min(3, channels)

        theta = image.new_empty((channels_to_gen, 1, 1)).uniform_(0, math.pi)
        center_y = image.new_empty((channels_to_gen, 1, 1)).uniform_(0, h)
        center_x = image.new_empty((channels_to_gen, 1, 1)).uniform_(0, w)

        y_grid = self._grid_y1.unsqueeze(0)
        x_grid = self._grid_x1.unsqueeze(0)
        dist = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        z1 = 0.5 + 0.5 * torch.cos(2 * math.pi * dist)

        phase = torch.cos(theta) * x_grid + torch.sin(theta) * y_grid
        z2 = 0.5 + 0.5 * torch.cos(phase)

        moire = (torch.minimum(z1, z2) + 1) / 2
        if channels > channels_to_gen:
            padded = image.new_zeros((channels, h, w))
            padded[:channels_to_gen] = moire
            moire = padded

        return moire.unsqueeze(0)

    def forward(self, image_and_cover):
        embed_image, cover_image = image_and_cover 
        self._prepare_cache(embed_image)
             
        # perspective transform
        noised_image = self._perspective(embed_image, 2)

        # Light Distortion
        c = random.randint(0, 1)
        L = self._light_distortion(c, embed_image)

        # Moire Distortion
        Z = self._moire_distortion(embed_image) * 2 - 1
        
        # Mingle Light and Moire
        noised_image = noised_image * L * 0.85 + Z * 0.15

        # Gaussian noise
        noised_image = noised_image + (0.001**0.5) * torch.randn_like(noised_image)

        return noised_image
