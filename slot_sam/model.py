import torch
from torch import nn

from mobile_sam.modeling import Sam
from slot_sam.slot_attn import SlotAttentionEncoder
from slot_sam.slot_decoder import PointDecoder


class SlotSam(nn.Module):
    def __init__(self, mobile_sam: Sam, num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size, num_heads,
                 weight_power):
        super().__init__()
        self.mobile_sam = mobile_sam
        self.slot_attention_encoder = SlotAttentionEncoder(
            num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size, num_heads, weight_power
        )
        self.slot_decoder = PointDecoder(slot_size=slot_size)

    def forward(self, image):
        image_embeddings = self.mobile_sam.image_encoder(image)
        batch_size, sam_channels, sam_h_enc, sam_w_enc = image_embeddings.size()
        image_embeddings = image_embeddings.flatten(start_dim=2).permute(0, 2, 1)

        slots, attention = self.slot_attention_encoder(image_embeddings)
        _, num_slots, slot_size = slots.size()
        points = self.slot_decoder(slots.view(batch_size * num_slots, slot_size))

        # point_dimension == 2 for 2D points
        _, point_dimension = points.size()
        points = points * (torch.as_tensor(
            [self.mobile_sam.prompt_encoder.input_image_size], dtype=torch.float32, device=slots.device
        ) - 1)
        points = points.view(batch_size, 1, num_slots, point_dimension)
        points = points.expand(-1, num_slots, -1, -1)
        points = points.reshape(batch_size * num_slots, num_slots, point_dimension)
        labels = torch.eye(num_slots, dtype=torch.int32, device=slots.device).unsqueeze(0)
        labels = labels.expand(batch_size, -1, -1).reshape(batch_size * num_slots, num_slots)
        prompt_encoder_input = (points, labels)
        sparse_embeddings, dense_embeddings = self.mobile_sam.prompt_encoder(
            points=prompt_encoder_input,
            boxes=None,
            masks=None,
        )

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
        return low_res_masks.view(batch_size, num_slots, mask_h, mask_w), slots, points[labels.to(torch.bool)].view(batch_size, num_slots, -1), attention
