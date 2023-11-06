from typing import Tuple, Union

import torch
from torch import nn, Tensor

from mobile_sam.modeling import Sam
from slot_sam.slot_attn import SlotAttentionEncoder
from slot_sam.slot_decoder import PointDecoder


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj


class SlotSam(nn.Module):
    def __init__(self, mobile_sam: Sam, num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size, num_heads,
                 weight_power):
        super().__init__()
        self.mobile_sam = mobile_sam
        self.slot_attention_encoder = SlotAttentionEncoder(
            num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size, num_heads, weight_power
        )
        self.slot_decoder = PointDecoder(slot_size=slot_size)

        hidden_dims = (64, 64, 64, 64)
        resolution = (64, 64)
        out_features = hidden_dims[-1]
        kernel_size = 5
        modules = []
        channels = 3
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(3, out_features, resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, image_upscale, image):
        batch_size = image.size()[0]
        encoder_out = self.encoder(image)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)

        slots, attention = self.slot_attention_encoder(encoder_out)
        _, num_slots, slot_size = slots.size()
        points = self.slot_decoder(slots.view(batch_size * num_slots, slot_size))
        points = points * (torch.as_tensor(
            [self.mobile_sam.prompt_encoder.input_image_size], dtype=torch.float32, device=slots.device
        ) - 1)
        labels = torch.ones(batch_size * num_slots, 1, dtype=torch.int32, device=slots.device)
        prompt_encoder_input = (points.unsqueeze(1), labels)
        sparse_embeddings, dense_embeddings = self.mobile_sam.prompt_encoder(
            points=prompt_encoder_input,
            boxes=None,
            masks=None,
        )

        image_embeddings = self.mobile_sam.image_encoder(image_upscale)
        batch_size, sam_channels, sam_h_enc, sam_w_enc = image_embeddings.size()
        image_embeddings = image_embeddings.flatten(start_dim=2).permute(0, 2, 1)
        image_embeddings = image_embeddings.permute(0, 2, 1).view(batch_size, sam_channels, sam_h_enc, sam_w_enc)
        image_embeddings = image_embeddings.unsqueeze(1).expand(-1, num_slots, -1, -1, -1)
        image_embeddings = image_embeddings.reshape(batch_size * num_slots, sam_channels, sam_h_enc, sam_w_enc)
        image_pe = self.mobile_sam.prompt_encoder.get_dense_pe()

        low_res_masks, iou_predictions = self.mobile_sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        _, _, mask_h, mask_w = low_res_masks.size()
        return low_res_masks.view(batch_size, num_slots, mask_h, mask_w), slots, points.view(batch_size, num_slots, -1), attention
