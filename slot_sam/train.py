import argparse
import collections

import gym
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import utils as vutils

import envs
from mobile_sam import sam_model_registry
from slot_sam.data import TransformSam, Shapes2dDataset
from slot_sam.model import SlotSam


def get_parameters(model):
    return [param for name, param in model.named_parameters() if not name.startswith('mobile_sam')]


def collect_data(dataset_size):
    env = gym.make('Navigation5x5-v0')
    observations = []
    for i in range(dataset_size):
        observations.append(env.reset())

    env.close()

    return np.asarray(observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sam_checkpoint_path', type=str, default='../weights/mobile_sam.pt')
    parser.add_argument('--sam_model_type', type=str, default='vit_t')
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--num_slots', type=int, default=6)
    parser.add_argument('--slot_size', type=int, default=64)
    parser.add_argument('--mlp_hidden_size', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--n_points', type=int, default=1)
    parser.add_argument('--weight_power', type=float, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=5)
    parser.add_argument('--n_visualize_images', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--intersection_loss_coef', type=float, default=10)
    parser.add_argument('--dataset_path', type=str, default='./shapes2d_dataset')
    parser.add_argument('--dataset_size', type=int, default=5000)
    parser.add_argument('--wandb_project', type=str, default='Test project')
    parser.add_argument('--wandb_run', type=str, default='run-0')
    parser.add_argument('--wandb_dir', type=str, default='./wandb')

    args = parser.parse_args()

    device = torch.device(args.device)

    mobile_sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint_path)
    mobile_sam = mobile_sam.to(device=device)
    for param in mobile_sam.parameters():
        param.requires_grad = False
    mobile_sam.eval()

    input_channels = 256
    slot_sam = SlotSam(
        mobile_sam,
        args.num_iterations,
        args.num_slots,
        input_channels,
        args.slot_size,
        args.mlp_hidden_size,
        args.num_heads,
        weight_power=args.weight_power
    )
    slot_sam = slot_sam.to(device)
    optimizer = torch.optim.Adam(get_parameters(slot_sam), lr=args.learning_rate)

    dataset = Shapes2dDataset(transform=TransformSam(), path=args.dataset_path, size=args.dataset_size)
    dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    dataset_size = len(dataset)

    run = wandb.init(
        project=args.wandb_project,
        config=vars(args),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        name=args.wandb_run,
        dir=args.wandb_dir
    )

    for epoch in range(args.n_epochs):
        record = collections.Counter()
        for batch_index, batch in enumerate(dataloader):
            image_torch = batch[0].to(device)
            low_res_masks, slots, points, attention = slot_sam(image_torch)
            low_res_masks_sum = low_res_masks.sum(dim=1)
            coverage_loss = torch.nn.functional.binary_cross_entropy_with_logits(low_res_masks_sum, torch.ones_like(low_res_masks_sum))
            intersection_loss = 0
            n_slots = slots.size()[1]
            for i in range(n_slots):
                for j in range(i + 1, n_slots):
                    intersection_loss += torch.mean(torch.sigmoid(torch.minimum(low_res_masks[:, i], low_res_masks[:, j])) ** 2)

            loss = coverage_loss + args.intersection_loss_coef * intersection_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(get_parameters(slot_sam), max_norm=args.max_grad_norm)
            optimizer.step()
            record['coverage_loss'] += coverage_loss.item()
            record['intersection_loss'] += intersection_loss.item()
            record['grad_norm'] += grad_norm.item()
        for key in record:
            record[key] /= dataset_size

        record['epoch'] = epoch

        val_images = torch.as_tensor(batch[1][:args.n_visualize_images]).permute(0, 3, 1, 2) / 255
        resize = transforms.Resize(size=val_images.size()[2:])
        log_images = []
        for val_image, val_points, masks in zip(val_images, points[:args.n_visualize_images].detach().cpu(), torch.sigmoid(low_res_masks[:2]).detach().cpu()):
            images = [val_image]
            for mask, point in zip(masks, val_points):
                resized_mask = resize(mask.unsqueeze(0))
                image_masked = val_image * resized_mask + (1 - resized_mask)
                point *= val_images.size()[-1] / slot_sam.mobile_sam.image_encoder.img_size
                point = point.to(torch.int64)
                image_masked[:, point[1] - 2: point[1] + 2, point[0] - 2: point[0] + 2] = torch.as_tensor([1., 0., 0.]).view(3, 1, 1)
                images.append(image_masked)
            log_images.append(torch.stack(images, dim=0))

        log_images = torch.stack(log_images, dim=0)
        nrows, ncols, c, h, w = log_images.size()
        log_images = vutils.make_grid(log_images.view(nrows * ncols, c, h, w), normalize=False, nrow=ncols)
        record['images'] = wandb.Image(log_images)
        wandb.log(record)

