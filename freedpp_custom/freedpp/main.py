#!/usr/bin/env python3
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger
from rdkit.Chem import Lipinski, MolFromSmiles
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchvision.ops import MLP

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from freedpp.args import parse_args, update_args
from freedpp.env.docking import DockingVina
from freedpp.env.environment import Environment
from freedpp.env.reward import (
    REWARD_COMMENTS,
    MolLogP,
    OutOfRange,
    PatternFilter,
    Reward,
    drug_likeness_reward,
    identity,
)
from freedpp.metrics import compute_metrics as _compute_metrics
from freedpp.train.nn import Actor, Critic, Encoder, Prioritizer
from freedpp.train.replay_buffer import ReplayBuffer
from freedpp.train.sac import SAC
from freedpp.train.utils import log_info
from freedpp.utils import CacheAndPool, dump2json, int2str, makedirs, read_mols, set_seed


RDLogger.DisableLog("rdApp.*")


def compute_metrics(args, epoch):
    mols = read_mols(args, epoch)
    metrics = _compute_metrics(
        mols,
        ref={},
        k=args["unique_k"],
        batch_size=args["batch_size"],
        n_jobs=args["n_jobs"],
    )
    suffix = int2str(epoch)
    dump2json(metrics, os.path.join(args["metrics_dir"], f"metrics_{suffix}.json"))


def evaluate(args, epoch, env):
    mols = read_mols(args, epoch)
    rewards = env.reward_batch(mols)
    comments = [REWARD_COMMENTS.get(smi, "") for smi in mols]
    score_values = rewards.get("DrugLikenessProperty") or rewards.get("Reward")
    suffix = int2str(epoch)
    log_info(
        os.path.join(args["mols_dir"], f"sample_{suffix}.csv"),
        {"Smiles": mols, "smiles": mols, "score": score_values, "comment": comments, **rewards},
        epoch,
        additional_info=None,
        writer=None,
    )


def init_rewards(args):
    assert len(args["objectives"]) == len(args["weights"])
    alert_table = pd.read_csv(args["alert_collections"])
    patterns = {}
    for name in ["PAINS", "SureChEMBL", "Glaxo"]:
        patterns[name] = alert_table[alert_table["rule_set_name"] == name]["smarts"]

    hard = args["reward_version"] == "hard"
    preprocess = MolFromSmiles
    costs = {
        "DockingScore": Reward(DockingVina(args["docking_config"]), partial(min, 0)),
        "LogP": Reward(MolLogP, OutOfRange(lower=0, upper=5, hard=hard), preprocess=preprocess),
        "HeavyAtomCount": Reward(Lipinski.HeavyAtomCount, OutOfRange(upper=40, hard=hard), preprocess=preprocess),
        "NumHAcceptors": Reward(Lipinski.NumHAcceptors, OutOfRange(upper=10, hard=hard), preprocess=preprocess),
        "NumHDonors": Reward(Lipinski.NumHDonors, OutOfRange(upper=5, hard=hard), preprocess=preprocess),
        "PAINS": Reward(PatternFilter(patterns["PAINS"]), identity, preprocess=preprocess),
        "SureChEMBL": Reward(PatternFilter(patterns["SureChEMBL"]), identity, preprocess=preprocess),
        "Glaxo": Reward(PatternFilter(patterns["Glaxo"]), identity, preprocess=preprocess),
        "DrugLikeness": Reward(drug_likeness_reward, identity, preprocess=preprocess),
    }

    rewards = {}
    for name, weight in zip(args["objectives"], args["weights"]):
        costs[name].weight = weight if name == "DrugLikeness" else -weight
        processes = args["num_sub_proc"] if name == "DockingScore" else 1
        rewards[name] = CacheAndPool(costs[name], processes=processes)

    return rewards


def init_models(args, env, checkpoint=None):
    device = args["device"]
    emb_size = s = args["emb_size"]
    num_fragments = len(env.fragments)
    mlp_kwargs = {"norm_layer": nn.LayerNorm}
    critic_args = (4 * s, (2 * s, s, s, 1))
    action_dim = num_fragments if args["action_mechanism"] == "pi" else 1
    actor_args = ((s, (s, s, 1)), (s, (s, s, action_dim)), (s, (s, s, 1)))
    actor_kwargs = ({}, {}, {})

    set_seed(args["seed"])
    critic_encoder = actor_encoder = Encoder(
        env.atom_dim,
        emb_size=emb_size,
        n_layers=args["n_layers"],
        aggregation=args["aggregation"],
    )
    critic = Critic(
        critic_encoder,
        env.fragments,
        emb_size=emb_size,
        n_nets=args["n_nets"],
        mlp_args=critic_args,
        mlp_kwargs=mlp_kwargs,
    ).to(device)
    set_seed(args["seed"])
    critic_target_encoder = Encoder(
        env.atom_dim,
        emb_size=emb_size,
        n_layers=args["n_layers"],
        aggregation=args["aggregation"],
    )
    critic_target = Critic(
        critic_target_encoder,
        env.fragments,
        emb_size=emb_size,
        n_nets=args["n_nets"],
        mlp_args=critic_args,
        mlp_kwargs=mlp_kwargs,
    ).to(device)
    actor = Actor(
        actor_encoder,
        env.fragments,
        emb_size=emb_size,
        tau=args["tau"],
        actions_dim=env.actions_dim,
        mlp_args=actor_args,
        mlp_kwargs=actor_kwargs,
        fragmentation=args["fragmentation"],
        merger=args["merger"],
        mechanism=args["action_mechanism"],
        ecfp_size=args["ecfp_size"],
    ).to(device)
    log_alpha = torch.tensor([np.log(args["alpha"])], requires_grad=args["train_alpha"], device=device)

    prioritizer = None
    if args["per"]:
        prioritizer_args = (s, (s, s, s, 1))
        prioritizer_encoder = Encoder(
            env.atom_dim,
            emb_size=emb_size,
            n_layers=args["n_layers"],
            aggregation=args["aggregation"],
        ).to(device)
        prioritizer_head = MLP(*prioritizer_args, **mlp_kwargs)
        prioritizer = Prioritizer(prioritizer_encoder, prioritizer_head).to(device)

    if checkpoint:
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        critic_target.load_state_dict(checkpoint["critic_target"])
        log_alpha = torch.tensor([checkpoint["log_alpha"]], requires_grad=args["train_alpha"], device=device)
        if prioritizer:
            prioritizer.load_state_dict(checkpoint["prioritizer"])

    return actor, critic, critic_target, log_alpha, prioritizer


def init_optimizers(actor, critic, critic_target, log_alpha, prioritizer, args, checkpoint=None):
    del critic_target
    actor_parameters = actor.parameters()
    encoder_parameters = actor.encoder.parameters()
    actor_parameters = [p for p in actor_parameters if p not in encoder_parameters]
    actor_optimizer = Adam(actor_parameters, lr=args["actor_lr"], weight_decay=args["weight_decay"])
    critic_optimizer = Adam(critic.parameters(), lr=args["critic_lr"], weight_decay=args["weight_decay"])
    alpha_optimizer = Adam([log_alpha], lr=args["alpha_lr"], eps=args["alpha_eps"])
    prioritizer_optimizer = None
    if prioritizer:
        prioritizer_optimizer = Adam(
            prioritizer.parameters(),
            lr=args["prioritizer_lr"],
            weight_decay=args["weight_decay"],
        )

    if checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if prioritizer:
            prioritizer_optimizer.load_state_dict(checkpoint["prioritizer_optimizer"])

    return actor_optimizer, critic_optimizer, alpha_optimizer, prioritizer_optimizer


def init_sac(args, env, checkpoint=None):
    epoch = 0
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        epoch = checkpoint["epoch"]

    models = init_models(args, env, checkpoint=checkpoint)
    optimizers = init_optimizers(*models, args, checkpoint=checkpoint)
    replay_buffer = ReplayBuffer(
        size=args["replay_size"],
        priority=bool(models[-1]),
        dzeta=args["dzeta"],
    )
    return models, optimizers, epoch, replay_buffer


def main(args):
    commands = args["commands"]
    args["rewards"] = init_rewards(args)
    env = Environment(**args)
    if "train" in commands or "sample" in commands:
        writer = SummaryWriter(args["logs_dir"])
        writer_id = ",".join(commands + [args["checkpoint"]])
        models, optimizers, epoch, replay_buffer = init_sac(args, env, checkpoint=args["checkpoint"])
        sac = SAC(*models, *optimizers, env, replay_buffer, writer, epoch=epoch, **args)
        writer.add_text(writer_id, str(args), epoch)
    else:
        epoch = args["epochs"]
    if "train" in commands:
        sac.train()
    if "sample" in commands:
        sac.sample(num_mols=args["num_mols"], dump=True)
    if "calc_metrics" in commands:
        compute_metrics(args, epoch)
    if "evaluate" in commands:
        evaluate(args, epoch, env)


def setup():
    args = parse_args()
    update_args(args)
    set_seed(args.seed)
    if os.path.exists(args.exp_dir) and "train" in args.commands and not args.checkpoint:
        raise ValueError(f'Experiment directory "{args.exp_dir}" already exists.')
    makedirs(args)
    return vars(args)


if __name__ == "__main__":
    main(setup())
