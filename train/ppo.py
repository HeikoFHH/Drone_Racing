"""Compact PPO with an asymmetric actor-critic (SIMULATION_PLAN.md section 12, D6).

Actor sees the realizable observation only; the critic additionally sees the privileged state
(``env._priv()``). Observation normalisation is frozen within an iteration and updated after it,
so logged log-probs match the ones used in the update.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape, device, eps=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, *self.mean.shape).float()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), torch.tensor(float(x.shape[0]), device=x.device)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    def normalize(self, x, clip=10.0):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-clip, clip)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count.cpu()}

    def load_state_dict(self, s, device):
        self.mean, self.var, self.count = (s[k].to(device) for k in ("mean", "var", "count"))


def mlp(inp, hidden, out):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim=4, hidden=(256, 256, 128), log_std_init=-0.7):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, act_dim)
        self.critic = mlp(obs_dim + priv_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        for m in self.actor[-1:]:
            nn.init.orthogonal_(m.weight, 0.01)

    def mean(self, obs):
        return torch.tanh(self.actor(obs))  # tanh-bounded action mean

    def dist(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs, priv):
        return self.critic(torch.cat([obs, priv], -1)).squeeze(-1)


@dataclass
class PPOCfg:
    horizon: int = 32
    epochs: int = 5
    minibatches: int = 8
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    kl_target: float = 0.01
    clip: float = 0.2
    value_clip: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    obs_clip: float = 10.0


class PPO:
    def __init__(self, env, cfg: PPOCfg | None = None, device="cpu", hidden=(256, 256, 128)):
        self.env, self.cfg, self.device = env, cfg or PPOCfg(), torch.device(device)
        self.net = ActorCritic(env.obs_dim, env.priv_dim, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.lr = self.cfg.lr
        self.obs_rms = RunningMeanStd((env.obs_dim,), self.device)
        self.priv_rms = RunningMeanStd((env.priv_dim,), self.device)
        self.ret_rms = RunningMeanStd((), self.device)
        self.disc_ret = torch.zeros(env.num_envs, device=self.device)
        self.obs, info = env.reset()
        self.priv = info["priv"]
        self.iteration = 0

    # -- helpers ----------------------------------------------------------------------------------
    def _n(self, obs, priv):
        c = self.cfg.obs_clip
        return self.obs_rms.normalize(obs, c), self.priv_rms.normalize(priv, c)

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.net.mean(self.obs_rms.normalize(obs.to(self.device), self.cfg.obs_clip))

    # -- rollout ----------------------------------------------------------------------------------
    @torch.no_grad()
    def collect(self):
        c, env, T, E = self.cfg, self.env, self.cfg.horizon, self.env.num_envs
        buf = {k: [] for k in ("obs", "priv", "act", "logp", "rew", "term", "val")}
        raw_obs, raw_priv = [], []
        stats = {"ep_return": [], "ep_len": [], "success": [], "lap_time": [], "cause": []}
        comp_sum: dict[str, float] = {}
        for _ in range(T):
            o, p = self._n(self.obs, self.priv)
            dist = self.net.dist(o)
            a = dist.sample()
            logp = dist.log_prob(a).sum(-1)
            val = self.net.value(o, p)
            nobs, rew, term, trunc, info = env.step(a.clamp(-1, 1))
            done = info["done"]
            r = rew.clone()
            if (trunc & done).any():  # bootstrap through time-outs
                m = trunc
                fo, fp = self._n(info["final_obs"], info["final_priv"])
                r[m] += c.gamma * self.net.value(fo, fp)[m]
            self.disc_ret = self.disc_ret * c.gamma + rew
            self.ret_rms.update(self.disc_ret)
            self.disc_ret[done] = 0.0
            for k, v in (("obs", o), ("priv", p), ("act", a), ("logp", logp), ("rew", r),
                         ("term", done.float()), ("val", val)):
                buf[k].append(v)
            raw_obs.append(self.obs)
            raw_priv.append(self.priv)
            if done.any():
                stats["ep_return"].append(info["ep_return"][done])
                stats["ep_len"].append(info["ep_len"][done].float())
                stats["success"].append((info["cause"][done] == 1).float())
                stats["cause"].append(info["cause"][done])
                lt = info["lap_time"][done]
                stats["lap_time"].append(lt[torch.isfinite(lt)])
            for k, v in info["reward_components"].items():
                comp_sum[k] = comp_sum.get(k, 0.0) + float(v.mean())
            self.obs, self.priv = nobs, info["priv"]
        o, p = self._n(self.obs, self.priv)
        last_val = self.net.value(o, p)
        for k in buf:
            buf[k] = torch.stack(buf[k])
        scale = torch.sqrt(self.ret_rms.var + 1e-8)
        buf["rew"] = buf["rew"] / scale
        adv = torch.zeros_like(buf["rew"])
        gae, nxt = torch.zeros(E, device=self.device), last_val
        for t in reversed(range(T)):
            nonterm = 1.0 - buf["term"][t]
            delta = buf["rew"][t] + c.gamma * nxt * nonterm - buf["val"][t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
            nxt = buf["val"][t]
        buf["adv"], buf["ret"] = adv, adv + buf["val"]
        self.obs_rms.update(torch.stack(raw_obs))
        self.priv_rms.update(torch.stack(raw_priv))
        return buf, stats, {k: v / T for k, v in comp_sum.items()}

    # -- update -----------------------------------------------------------------------------------
    def update(self, buf):
        c = self.cfg
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in buf.items()}
        n = flat["obs"].shape[0]
        adv = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        mb = n // c.minibatches
        kls, pl, vl, ent = [], [], [], []
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(c.minibatches):
                ix = perm[i * mb:(i + 1) * mb]
                dist = self.net.dist(flat["obs"][ix])
                logp = dist.log_prob(flat["act"][ix]).sum(-1)
                ratio = (logp - flat["logp"][ix]).exp()
                a = adv[ix]
                p_loss = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                v = self.net.value(flat["obs"][ix], flat["priv"][ix])
                v_old, ret = flat["val"][ix], flat["ret"][ix]
                v_clip = v_old + (v - v_old).clamp(-c.value_clip, c.value_clip)
                v_loss = 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = p_loss + c.vf_coef * v_loss - c.ent_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kls.append((flat["logp"][ix] - logp).mean())
                    pl.append(p_loss.detach()); vl.append(v_loss.detach()); ent.append(entropy.detach())
            kl = float(torch.stack(kls[-c.minibatches:]).mean())
            if kl > 2 * c.kl_target:
                self.lr = max(c.lr_min, self.lr / 1.5)
            elif kl < c.kl_target / 2:
                self.lr = min(c.lr_max, self.lr * 1.5)
            for g in self.opt.param_groups:
                g["lr"] = self.lr
        return {"kl": float(torch.stack(kls).mean()), "policy_loss": float(torch.stack(pl).mean()),
                "value_loss": float(torch.stack(vl).mean()), "entropy": float(torch.stack(ent).mean()), "lr": self.lr}

    def iterate(self):
        t0 = time.perf_counter()
        buf, stats, comps = self.collect()
        t1 = time.perf_counter()
        upd = self.update(buf)
        t2 = time.perf_counter()
        self.iteration += 1
        cat = lambda k: torch.cat(stats[k]) if stats[k] else torch.zeros(0, device=self.device)
        ep_r, ep_l, succ, lap, cause = cat("ep_return"), cat("ep_len"), cat("success"), cat("lap_time"), cat("cause")
        log = {"iter": self.iteration, "episodes": int(ep_r.numel()),
               "ep_return": float(ep_r.mean()) if ep_r.numel() else float("nan"),
               "ep_len_s": float(ep_l.mean()) * self.env.cfg.dt if ep_l.numel() else float("nan"),
               "success": float(succ.mean()) if succ.numel() else float("nan"),
               "lap_time_s": float(lap.median()) if lap.numel() else float("nan"),
               "log_std": float(self.net.log_std.mean()), "steps_per_s": self.cfg.horizon * self.env.num_envs / (t1 - t0),
               "collect_s": t1 - t0, "update_s": t2 - t1, **upd}
        if cause.numel():
            for code, name in ((2, "pole"), (3, "ground"), (4, "ceiling"), (5, "oob"), (6, "timeout")):
                log[f"cause_{name}"] = float((cause == code).float().mean())
        log.update({f"rew_{k}": v for k, v in comps.items()})
        return log

    # -- checkpoints ------------------------------------------------------------------------------
    def state_dict(self):
        return {"net": self.net.state_dict(), "obs_rms": self.obs_rms.state_dict(), "priv_rms": self.priv_rms.state_dict(),
                "iteration": self.iteration}

    def load_state_dict(self, s):
        self.net.load_state_dict(s["net"])
        self.obs_rms.load_state_dict(s["obs_rms"], self.device)
        self.priv_rms.load_state_dict(s["priv_rms"], self.device)
        self.iteration = s.get("iteration", 0)
