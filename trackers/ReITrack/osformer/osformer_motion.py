import os
import yaml
import torch


class MotionEvaluator:
    def __init__(self, motion_model_cfg, motion_epoch, dataloader=None, device="cuda:0"):
        self.dataloader = dataloader
        self.device = device
        self.load_model(motion_model_cfg, motion_epoch)

    def load_model(self, motion_model_cfg, motion_epoch):
        from trackers.ReITrack.osformer.models import OneStepFormer as MotionModel
        with open(motion_model_cfg, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        motion_model_dir = cfg["Train"]["output_dir"]
        motion_model_path = os.path.join(motion_model_dir, f"{motion_epoch}_ckpt.pth.tar")

        model_params = {
            'seq_len': cfg["Motion Model"]["seq_len"],
            'pred_len': cfg["Motion Model"]["pred_len"],
            'in_dim': cfg["Motion Model"]["in_dim"],
            'mhsa_dim': cfg["Motion Model"]["mhsa_dim"],
            'mhsa_d_ff': cfg["Motion Model"]["mhsa_d_ff"],
            'num_heads': cfg["Motion Model"]["num_heads"],
            'dropout': cfg["Motion Model"]["dropout"],
            'activation': cfg["Motion Model"]["activation"],
            'e_layers': cfg["Motion Model"]["e_layers"],
            'd_layers': cfg["Motion Model"]["d_layers"],
            'motion_out_dim': cfg["Motion Model"]["motion_out_dim"],
            'exp_v': cfg["Train"]["exp_v"]
        }

        self.model = MotionModel(model_params)
        ckpt = torch.load(motion_model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()


def get_motion_model(motion_model_cfg, motion_epoch):
    model_loader = MotionEvaluator(motion_model_cfg, motion_epoch)
    motion_model = model_loader.model
    return motion_model


def avg_vel(sequence):
    """

    :param sequence: (1, seq_len, 4)
    :return:
    """
    seq_vel = (sequence[:, 1:] - sequence[:, :-1]).permute(0, 2, 1)  # (1, 4, seq_len)
    tri_mat = torch.triu(torch.ones((seq_vel.shape[-1], seq_vel.shape[-1])), diagonal=0)
    avg_seq_vel = torch.matmul(seq_vel, tri_mat) / torch.arange(1, seq_vel.shape[-1] + 1).unsqueeze(0)
    return avg_seq_vel.permute(0, 2, 1)


def linear_motion_pred(sequences, pred_len, linear_center_only=True):
    """

    :param sequences: Tensor(N, seq_len, 4)
    :param pred_len:
    :return:
        pred_seq: Tensor(N, pred_len, 4)
    """
    vels = sequences[:, 1:, :] - sequences[:, :-1, :]
    num_estimates = vels.shape[1] - torch.isnan(vels).sum(1)
    num_estimates = num_estimates.min(-1)[0]
    vels = torch.nan_to_num(vels, nan=0.0)
    avg_vel = vels.sum(dim=1) / num_estimates[:, None]  # (N, 4)
    avg_vel = torch.nan_to_num(avg_vel, nan=0.0)

    if linear_center_only:
        avg_vel[..., 2:] = 0

    # Prediction
    timesteps = torch.arange(start=1, end=pred_len + 1, device=sequences.device)
    displace = torch.einsum('nd,t->ntd', avg_vel, timesteps)
    preds = displace + sequences[:, -1:, :]
    return preds
