import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
import torch.nn.functional as F
import os
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace("/registration", ""))
from environment import environment as env
from environment import transformations as tra
from environment.buffer import Buffer
from registration.model import Agent
import registration.model as util_model
import utility.metrics as metrics
from utility.logger import Logger
from dataset.dataset import DatasetModelnet40, DatasetLinemod
import config as cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(agent, logger, dataset, noise_type, epochs, lr, lr_step, alpha, model_path, reward_mode=""):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, 0.5)

    Dataset = DatasetModelnet40 if dataset == "m40" else DatasetLinemod
    train_dataset = Dataset("train", noise_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_dataset = Dataset("val", noise_type)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_dataset = Dataset("test" if dataset == "m40" else "eval", noise_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    RANDOM_STATE = np.random.get_state()  # otherwise loader produces deterministic samples after iter 1
    losses_bc, losses_ppo, train_rewards, final_rewards = [], [], [], []
    episode = 0  # for loss logging (not using epoch)
    best_chamfer = np.infty

    buffer = Buffer()
    buffer.start_trajectory()
    for epoch in range(epochs):
        print(f"Epoch {epoch}")

        # -- train
        agent.train()
        np.random.set_state(RANDOM_STATE)

        progress = tqdm(BackgroundGenerator(train_loader), total=len(train_loader))
        for data in progress:
            with torch.no_grad():
                # per sample, generate a full trajectory
                source, target, pose_source, pose_target = env.init(data)

                if cfg.DISENTANGLED:
                    pose_target = tra.to_disentangled(pose_target, source)
                current_source = source
                if reward_mode == "goal":
                    reward = env.reward_goal(pose_source, pose_target)
                elif reward_mode == "step":
                    gt_pcd_source = tra.apply_trafo(current_source, pose_target, disentangled=cfg.DISENTANGLED)
                    _, prev_chamfer = env.reward_step(current_source, gt_pcd_source)

                # STAGE 1: generate trajectories
                for step in range(cfg.ITER_TRAIN):
                    # expert prediction
                    expert_action = env.expert(pose_source, pose_target, mode=cfg.EXPERT_MODE)

                    # student prediction -- stochastic policy
                    state_emb, action_logit, state_value, _ = agent(current_source, target)

                    action = util_model.action_from_logits(action_logit, deterministic=False)
                    action_logprob, action_entropy = util_model.action_stats(action_logit, action)

                    # step environment and get reward
                    new_source, pose_source = env.step(source, action, pose_source, cfg.DISENTANGLED)
                    if reward_mode == "goal":
                        reward = env.reward_goal(pose_source, pose_target)
                    elif reward_mode == "step":
                        reward, prev_chamfer = env.reward_step(new_source, gt_pcd_source, prev_chamfer)
                    else:
                        reward = torch.zeros((pose_source.shape[0], 1, 1)).to(DEVICE)

                    # log trajectory
                    buffer.log_step([current_source, target], state_value, reward,
                                    expert_action,
                                    action, action_logit, action_logprob)

                    current_source = new_source

                    train_rewards.append(reward.view(-1))
                final_rewards.append(reward.view(-1))

            if len(buffer) == cfg.NUM_TRAJ:
                # STAGE 2: policy (and value estimator) update using BC (and PPO)

                # convert buffer to tensor of samples (also computes return and advantage over trajectories)
                samples = buffer.get_samples()
                ppo_dataset = torch.utils.data.TensorDataset(*samples)
                ppo_loader = torch.utils.data.DataLoader(ppo_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                                         drop_last=False)

                # sample batches from buffer and update
                for batch in ppo_loader:
                    sources, targets, \
                    expert_actions, state_values, \
                    actions, action_logits, action_logprobs, \
                    returns, advantages = batch

                    # -- predict using current policy
                    new_state_emb, new_action_logit, new_values, _ = agent(sources, targets)
                    new_action_logprob, new_action_entropy = util_model.action_stats(new_action_logit, actions)

                    # -- clone term
                    loss_translation = F.cross_entropy(new_action_logit[0].view(-1, 11, 1, 1, 1),
                                                       expert_actions[:, 0].reshape(-1, 1, 1, 1))
                    loss_rotation = F.cross_entropy(new_action_logit[1].view(-1, 11, 1, 1, 1),
                                                    expert_actions[:, 1].reshape(-1, 1, 1, 1))
                    clone_loss = (loss_translation + loss_rotation) / 2

                    if alpha > 0:
                        # -- policy term
                        # ratio: lp > prev_lp --> probability of selecting that action increased
                        ratio = torch.exp(new_action_logprob - action_logprobs).view(-1, 6)
                        policy_loss = -torch.min(ratio * advantages.repeat(1, 6),
                                                 ratio.clamp(1 - cfg.CLIP_EPS,
                                                             1 + cfg.CLIP_EPS) * advantages.repeat(1, 6)).mean()

                        # -- value term
                        value_loss = (new_values.view(-1, 1) - returns).pow(2)
                        if cfg.CLIP_VALUE:
                            values_clipped = state_values + (new_values - state_values)\
                                .clamp(-cfg.CLIP_EPS, cfg.CLIP_EPS)
                            losses_v_clipped = (values_clipped.view(-1, 1) - returns).pow(2)
                            value_loss = torch.max(value_loss, losses_v_clipped)
                        value_loss = value_loss.mean()

                        # -- entropy term
                        entropy_loss = new_action_entropy.mean()

                    # -- update
                    optimizer.zero_grad()
                    loss = clone_loss
                    losses_bc.append(clone_loss.item())
                    if alpha > 0:
                        ppo_loss = policy_loss + value_loss * cfg.C_VALUE - entropy_loss * cfg.C_ENTROPY
                        loss += ppo_loss * alpha
                        losses_ppo.append(ppo_loss.item())
                    loss.backward()
                    optimizer.step()

                # logging
                if alpha > 0:
                    logger.record("train/ppo", np.mean(losses_ppo))
                logger.record("train/bc", np.mean(losses_bc))
                logger.record("train/reward", float(torch.cat(train_rewards, dim=0).mean()))
                logger.record("train/final_reward", float(torch.cat(final_rewards, dim=0).mean()))
                logger.dump(step=episode)

                # reset
                losses_bc, losses_ppo, train_rewards, final_rewards = [], [], [], []
                buffer.clear()
                episode += 1

            buffer.start_trajectory()
        scheduler.step()
        RANDOM_STATE = np.random.get_state()  # evaluation sets seeds again -- keep random state of the training stage

        # -- test
        if val_loader is not None:
            chamfer_val = evaluate(agent, logger, val_loader, prefix='val')
        if test_loader is not None:
            chamfer_test = evaluate(agent, logger, test_loader)

        if chamfer_test <= best_chamfer:
            print(f"new best: {chamfer_test}")
            best_chamfer = chamfer_test
            infos = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict()
            }
            util_model.save(agent, f"{model_path}.zip", infos)
        logger.dump(step=epoch)


def evaluate(agent, logger, loader, prefix='test'):
    agent.eval()
    progress = tqdm(BackgroundGenerator(loader), total=len(loader))
    predictions = []
    val_losses = []
    with torch.no_grad():
        for data in progress:
            source, target, pose_source, pose_target = env.init(data)
            if cfg.DISENTANGLED:
                pose_target = tra.to_disentangled(pose_target, source)

            current_source = source
            for step in range(cfg.ITER_EVAL):
                expert_action = env.expert(pose_source, pose_target, mode=cfg.EXPERT_MODE)

                state_emb, action_logit, _, _ = agent(current_source, target)
                action = util_model.action_from_logits(action_logit, deterministic=True)

                loss_translation = F.cross_entropy(action_logit[0].view(-1, 11, 1, 1, 1),
                                                   expert_action[:, 0].reshape(-1, 1, 1, 1))
                loss_rotation = F.cross_entropy(action_logit[1].view(-1, 11, 1, 1, 1),
                                                expert_action[:, 1].reshape(-1, 1, 1, 1))
                val_losses.append((loss_translation + loss_rotation).item()/2)

                current_source, pose_source = env.step(source, action, pose_source, cfg.DISENTANGLED)
            if cfg.DISENTANGLED:
                pose_source = tra.to_global(pose_source, source)
            predictions.append(pose_source)

    predictions = torch.cat(predictions)
    _, summary_metrics = metrics.compute_stats(predictions, data_loader=loader)

    # log test metrics
    if isinstance(loader.dataset, DatasetLinemod):
        logger.record(f"{prefix}/add", summary_metrics['add'])
        logger.record(f"{prefix}/adi", summary_metrics['adi'])
        return summary_metrics['add']
    else:
        logger.record(f"{prefix}/mae-r", summary_metrics['r_mae'])
        logger.record(f"{prefix}/mae-t", summary_metrics['t_mae'])
        logger.record(f"{prefix}/iso-r", summary_metrics['r_iso'])
        logger.record(f"{prefix}/iso-t", summary_metrics['t_iso'])
        logger.record(f"{prefix}/chamfer", summary_metrics['chamfer_dist'])
        logger.record(f"{prefix}/adi-auc", summary_metrics['adi_auc10'] * 100)
        return summary_metrics['chamfer_dist']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReAgent - training on ModelNet40 and LINEMOD')
    parser.add_argument('--mode', type=str, default='il', choices=['pretrain', 'il', 'ilrl'],
                        help='pretraining (pretrain), IL-only (il), IL+RL with a step-wise reward (ilrls).')
    parser.add_argument('--dataset', type=str, default='m40', choices=['m40', 'lm'],
                        help='Dataset used for training. All experiments on ModelNet40 and ScanObjectNN use the same '
                             'weights - train both with "m40". Experiments on LINEMOD ("lm") use no pretraining.')
    args = parser.parse_args()

    # PATHS
    dataset = args.dataset
    mode = args.mode
    code_path = os.path.dirname(os.path.abspath(__file__)).replace("/registration", "")
    if not os.path.exists(os.path.join(code_path, "logs")):
        os.mkdir(os.path.join(code_path, "logs"))
    if not os.path.exists(os.path.join(code_path, "weights")):
        os.mkdir(os.path.join(code_path, "weights"))
    model_path = os.path.join(code_path, f"weights/{dataset}_{mode}")
    logger = Logger(log_dir=os.path.join(code_path, f"logs/{dataset}/"), log_name=f"{mode}",
                    reset_num_timesteps=True)

    # TRAINING
    agent = Agent().to(DEVICE)

    if args.mode == "pretrain" and dataset == "m40":
        print(f"Training: dataset '{dataset}'  - mode '{args.mode}'")
        train(agent, logger, dataset, noise_type="clean", epochs=50, lr=1e-3, lr_step=10, alpha=0,
              model_path=model_path)
    else:
        if args.mode == "il":
            alpha = 0.0
            reward_mode = ""
        elif args.mode == "ilrl":
            alpha = 2.0 if dataset == "m40" else 0.1  # reduced influence on lm
            reward_mode = "step"
        else:
            raise ValueError("No pretraining on LINEMOD. Use 'il' or 'ilrl' instead.")
        print(f"Training: dataset '{dataset}' - mode '{args.mode}'{f' - alpha={alpha}' if args.mode != 'il' else ''}")

        if dataset == "m40":
            print("  loading pretrained weights...")
            if os.path.exists(os.path.join(code_path, f"weights/m40_pretrain.zip")):
                util_model.load(agent, os.path.join(code_path, f"weights/m40_pretrain.zip"))
            else:
                raise FileNotFoundError(f"No pretrained weights found at "
                                        f"{os.path.join(code_path, f'weights/m40_pretrain.zip')}. Run with "
                                        f"'pretrain' first or download the provided weights.")

        noise_type = "jitter" if dataset == "m40" else "segmentation"
        epochs = 50 if dataset == "m40" else 100
        lr = 1e-4 if dataset == "m40" else 1e-3
        lr_step = 10 if dataset == "m40" else 20

        train(agent, logger, dataset, noise_type, epochs=epochs, lr=lr, lr_step=lr_step,
              alpha=alpha, reward_mode=reward_mode, model_path=model_path)
