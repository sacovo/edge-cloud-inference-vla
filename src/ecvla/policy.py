"""EXP-0001 fast policy: one architecture family, scaled by width and depth to a target head size (README §3).

Per control step (20 Hz):
  one encoder per modality (proprio, torque window, F/T window, optional frozen-vision features), summed
  -> FiLM from the pooled conditioning -> temporal core -> cross-attention over the conditioning tokens
  -> action-chunk head (chunk x 7; default CHUNK = 8). At run time only the first action of each chunk is executed.

Swappable parts (fixed for the whole grid after the A1 architecture check; see the EXP-0001 change log):
- core "gru": GRU (the pre-registered default); "tf": causal transformer over the last WINDOW steps (ALiBi bias)
- head "l1": deterministic chunk regression with L1 loss; "flow": conditional flow matching over the chunk,
  sampled with FLOW_STEPS Euler steps

Conditioning tokens: a learned null token (always present), a task token (task index embedding) and, for state
conditions, one token per object slot built from [pos, quat, gripper contact] (+ latent age) plus an object-name
embedding, and (when the scene has them) one token per articulated fixture joint from [joint position, gripper
contact with the fixture] plus a joint-name embedding. Latent conditions (z from a slow model, e.g. `z-teacher`)
add one token from [z, z's age in seconds] next to the task token. Slots whose latest update has not arrived yet are masked out. The z conditions (A2) will add their
latent as further tokens through the same path.

Head size = all trainable parameters except the vision projection, which is reported separately.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ecvla.sensors import FT_DIM, PROPRIO_DIM, TORQUE_DIM

CHUNK = 8  # default chunk length (EXP-0001); PolicyConfig.chunk overrides it (EXP-0002 teachers: 40)
ACT_DIM = 7
OBJ_DIM = 8  # pos(3) + quat(4) + gripper contact(1)
WINDOW = 16  # transformer context, control steps (0.8 s)
FLOW_STEPS = 10
COND_KINDS = ("none", "task", "state", "latent")


@dataclass
class PolicyConfig:
    d: int
    layers: int = 1
    cond: str = "none"  # none | task | state | latent (state and latent include the task token)
    n_tasks: int = 1
    n_names: int = 1
    vision_dim: int = 0  # 0 = blind
    age_input: bool = False
    fixtures: bool = False  # fixture-joint tokens (state conditions in scenes with drawers / knobs)
    latent_dim: int = 0  # z size for latent conditions (its age is always appended)
    chunk: int = CHUNK  # predicted action-chunk length
    track: bool = False  # EXP-0003: z is the age-aligned plan window; output = plan + bounded residual
    track_bound: tuple = ()  # per action dim, normalized units; 0 = unbounded (EXP-0003: gripper)
    core: str = "gru"  # gru | tf
    head: str = "l1"  # l1 | flow

    def __post_init__(self):
        assert self.cond in COND_KINDS, self.cond
        assert self.core in ("gru", "tf") and self.head in ("l1", "flow"), (self.core, self.head)
        if self.track:
            assert self.cond == "latent" and self.head == "l1" and self.latent_dim == self.chunk * ACT_DIM
            assert len(self.track_bound) == ACT_DIM
            self.track_bound = tuple(float(x) for x in self.track_bound)

    @property
    def heads(self) -> int:
        return 1 if self.d < 128 else self.d // 64


class TransformerCore(nn.Module):
    """Causal pre-LN transformer; each step attends to itself and the previous WINDOW-1 steps.

    The recurrent state is the buffer of the last layers*(WINDOW-1) inputs (the receptive field), so stepwise use
    matches the sequence pass exactly.
    """

    def __init__(self, d: int, layers: int, heads: int):
        super().__init__()
        self.heads = heads
        self.context = layers * (WINDOW - 1)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), o=nn.Linear(d, d),
                ln2=nn.LayerNorm(d), mlp=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)),
            ))
            for _ in range(layers)
        )
        self.register_buffer("slopes", 2.0 ** (-8.0 * (torch.arange(heads) + 1) / heads), persistent=False)

    def forward(self, x, state=None):
        past = 0 if state is None else state.shape[1]
        if state is not None:
            x = torch.cat([state, x], 1)
        B, T, d = x.shape
        i = torch.arange(T, device=x.device)
        dist = (i[:, None] - i[None, :]).float()
        allowed = (dist >= 0) & (dist < WINDOW)
        bias = (-self.slopes[:, None, None] * dist).masked_fill(~allowed, float("-inf"))
        h = x
        H = self.heads
        for blk in self.blocks:
            q, k, v = blk["qkv"](blk["ln1"](h)).view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
            h = h + blk["o"](a.transpose(1, 2).reshape(B, T, d))
            h = h + blk["mlp"](blk["ln2"](h))
        return h[:, past:], x[:, -self.context :].detach()


class FlowHead(nn.Module):
    """Velocity field v(a_tau, tau | f) for flow matching from Gaussian noise (tau=0) to the action chunk (tau=1)."""

    def __init__(self, d: int, chunk: int = CHUNK):
        super().__init__()
        self.chunk = chunk
        n = chunk * ACT_DIM
        self.net = nn.Sequential(nn.Linear(n + 1 + d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, n))

    def velocity(self, a, tau, f):
        return self.net(torch.cat([a, tau, f], -1))

    def loss(self, f, target):
        """Per-element squared velocity error, shaped like target (..., chunk, 7)."""
        a1 = target.flatten(-2)
        a0 = torch.randn_like(a1)
        tau = torch.rand(*a1.shape[:-1], 1, device=a1.device, dtype=a1.dtype)
        v = self.velocity((1 - tau) * a0 + tau * a1, tau, f)
        return (v - (a1 - a0)).pow(2).view_as(target)

    @torch.no_grad()
    def sample(self, f, generator=None):
        a = torch.randn(*f.shape[:-1], self.chunk * ACT_DIM, device=f.device, dtype=f.dtype, generator=generator)
        for s in range(FLOW_STEPS):
            tau = torch.full((*f.shape[:-1], 1), s / FLOW_STEPS, device=f.device, dtype=f.dtype)
            a = a + self.velocity(a, tau, f) / FLOW_STEPS
        return a.view(*f.shape[:-1], self.chunk, ACT_DIM)


class FastPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.enc_proprio = nn.Linear(PROPRIO_DIM, d)
        self.enc_torque = nn.Linear(TORQUE_DIM + 1, d)  # + fast_valid
        self.enc_ft = nn.Linear(FT_DIM + 1, d)
        self.vis_proj = nn.Linear(cfg.vision_dim, d) if cfg.vision_dim else None
        if cfg.cond != "none":
            self.null_tok = nn.Parameter(torch.zeros(d))
            self.task_emb = nn.Embedding(cfg.n_tasks, d)
            self.film = nn.Linear(d, 2 * d)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.q = nn.Linear(d, d)
            self.kv = nn.Linear(d, 2 * d)
            self.o = nn.Linear(d, d)
        if cfg.cond == "state":
            self.enc_obj = nn.Linear(OBJ_DIM + int(cfg.age_input), d)
            self.name_emb = nn.Embedding(cfg.n_names, d)
            if cfg.fixtures:
                self.enc_fix = nn.Linear(2 + int(cfg.age_input), d)
        if cfg.cond == "latent":
            assert cfg.latent_dim > 0
            self.enc_lat = nn.Linear(cfg.latent_dim + 1, d)
        if cfg.core == "gru":
            self.core = nn.GRU(d, d, num_layers=cfg.layers, batch_first=True)
        else:
            self.core = TransformerCore(d, cfg.layers, cfg.heads)
        self.head = nn.Linear(d, cfg.chunk * ACT_DIM) if cfg.head == "l1" else FlowHead(d, cfg.chunk)
        if cfg.track:  # residual starts at zero: a fresh plan is executed as is
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.register_buffer("bound", torch.tensor(cfg.track_bound), persistent=False)

    def head_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("vis_proj"))

    def vision_params(self) -> int:
        return sum(p.numel() for p in self.vis_proj.parameters()) if self.vis_proj is not None else 0

    def cond_tokens(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens (B, T, N, d) and key mask (B, T, N), True = attend."""
        B, T = b["proprio"].shape[:2]
        d = self.cfg.d
        task = self.task_emb(b["task"])[:, None, None].expand(B, T, 1, d)
        toks = [self.null_tok.expand(B, T, 1, d), task]
        mask = [torch.ones(B, T, 2, dtype=torch.bool, device=task.device)]
        if self.cfg.cond == "state":
            obj = torch.cat([b["obj_pose"], b["contact"][..., None]], -1)  # (B, T, S, 8)
            if self.cfg.age_input:
                obj = torch.cat([obj, b["age"][:, :, None, None].expand(*obj.shape[:3], 1)], -1)
            toks.append(self.enc_obj(obj) + self.name_emb(b["obj_name"])[:, None])
            mask.append(b["obj_mask"][:, None, :] & b["state_avail"][:, :, None])
            if self.cfg.fixtures:
                fix = b["fix_state"]  # (B, T, J, 2)
                if self.cfg.age_input:
                    fix = torch.cat([fix, b["age"][:, :, None, None].expand(*fix.shape[:3], 1)], -1)
                toks.append(self.enc_fix(fix) + self.name_emb(b["fix_name"])[:, None])
                mask.append(b["fix_mask"][:, None, :] & b["state_avail"][:, :, None])
        if self.cfg.cond == "latent":
            z = torch.cat([b["latent"], b["age"][..., None]], -1)  # (B, T, latent_dim + 1)
            toks.append(self.enc_lat(z)[:, :, None])
            mask.append(b["state_avail"][:, :, None])
        return torch.cat(toks, 2), torch.cat(mask, 2)

    def features(self, b: dict, state=None) -> tuple[torch.Tensor, object]:
        """b: per-step inputs (B, T, ...) plus per-episode `task`, `obj_name`, `obj_mask`. Returns (B, T, d), core state."""
        valid = b["fast_valid"][..., None]
        x = self.enc_proprio(b["proprio"]) + self.enc_torque(torch.cat([b["torque"], valid], -1))
        x = x + self.enc_ft(torch.cat([b["ft"], valid], -1))
        if self.vis_proj is not None:
            x = x + self.vis_proj(b["vision"])
        x = F.gelu(x)
        if self.cfg.cond != "none":
            toks, mask = self.cond_tokens(b)
            m = mask[..., None].to(toks.dtype)
            pooled = (toks * m).sum(2) / m.sum(2)
            gamma, beta = self.film(pooled).chunk(2, -1)
            x = x * (1 + gamma) + beta
        h, state = self.core(x, state)
        if self.cfg.cond != "none":
            h = h + self.attend(h, toks, mask)
        return F.gelu(h), state

    def attend(self, h, toks, mask):
        B, T, N, d = toks.shape
        H = self.cfg.heads
        q = self.q(h).view(B, T, H, d // H)
        k, v = self.kv(toks).view(B, T, N, 2, H, d // H).unbind(3)
        att = torch.einsum("bthc,btnhc->bthn", q, k) / math.sqrt(d // H)
        att = att.masked_fill(~mask[:, :, None, :], float("-inf")).softmax(-1)
        return self.o(torch.einsum("bthn,btnhc->bthc", att, v).reshape(B, T, d))

    def track_out(self, r: torch.Tensor, b: dict) -> torch.Tensor:
        """EXP-0003: plan window (b["latent"], normalized action units) + residual, bounded per dim as
        bound * tanh(r / bound) (0 = unbounded). Before the first plan arrives: the raw residual, no base."""
        bound = self.bound.to(r.dtype)
        rb = torch.where(bound > 0, bound * torch.tanh(r / bound.clamp_min(1e-6)), r)
        base = b["latent"].view_as(r).to(r.dtype)
        return torch.where(b["state_avail"][..., None, None], base + rb, r)

    def loss(self, f: torch.Tensor, target: torch.Tensor, b: dict | None = None) -> torch.Tensor:
        """Per-element loss (B, T, chunk, 7) on normalized actions; the caller masks and averages."""
        if self.cfg.head == "l1":
            out = self.head(f).view_as(target)
            return ((self.track_out(out, b) if self.cfg.track else out) - target).abs()
        return self.head.loss(f, target)

    @torch.no_grad()
    def act(self, f: torch.Tensor, generator=None, b: dict | None = None) -> torch.Tensor:
        """Normalized action chunk (B, T, chunk, 7). `track` models need the batch b (plan window, availability)."""
        if self.cfg.head == "l1":
            out = self.head(f).view(*f.shape[:-1], self.cfg.chunk, ACT_DIM)
            return self.track_out(out, b) if self.cfg.track else out
        return self.head.sample(f, generator)


def plan_window(z: torch.Tensor, age_steps: torch.Tensor, window: int) -> torch.Tensor:
    """EXP-0003: steps age .. age + window - 1 of a flattened plan z (..., H * 7), clamped to its last step.

    z: (..., H*7); age_steps: (...) integer steps since the plan was made. Returns (..., window * 7)."""
    H = z.shape[-1] // ACT_DIM
    j = (age_steps.long()[..., None] + torch.arange(window, device=z.device)).clamp(0, H - 1)  # (..., window)
    plan = z.view(*z.shape[:-1], H, ACT_DIM)
    return torch.gather(plan, -2, j[..., None].expand(*j.shape, ACT_DIM)).flatten(-2)


def count_head_params(cfg: PolicyConfig) -> int:
    with torch.device("meta"):
        return FastPolicy(cfg).head_params()


def config_for_size(target: int, **kw) -> PolicyConfig:
    """Width (and depth: 1 core layer below 1M params, else 2) whose head size is closest to `target` in log space."""
    layers = 1 if target <= 1_000_000 else 2
    best = None
    for d in range(4, 4097, 4):
        if d >= 128 and d % 64:
            continue  # wide models: attention heads of 64 channels
        cfg = PolicyConfig(d=d, layers=layers, **kw)
        n = count_head_params(cfg)
        err = abs(math.log(n / target))
        if best is None or err < best[0]:
            best = (err, cfg)
        if n > target:
            break
    return best[1]


def config_dict(cfg: PolicyConfig) -> dict:
    return asdict(cfg)
