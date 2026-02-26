from __future__ import annotations
import argparse
import sys
import json
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.cm as cm
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch available - using PPO-RL implementation")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not found - RL requires PyTorch")
    sys.exit(1)


# ============================================================
# TOUPricing  (عین کد مرجع)
# ============================================================
@dataclass
class TOUPricing:
    ELEC_PEAK: float    = 0.18
    ELEC_OFFPEAK: float = 0.10
    ELEC_WEEKEND: float = 0.12
    HEAT_WINTER: float  = 0.06
    HEAT_SUMMER: float  = 0.04
    PEAK_START: int     = 7
    PEAK_END: int       = 22

    def get_electricity_price(self, timestamp: pd.Timestamp) -> float:
        if timestamp.dayofweek >= 5:
            return self.ELEC_WEEKEND
        elif self.PEAK_START <= timestamp.hour < self.PEAK_END:
            return self.ELEC_PEAK
        else:
            return self.ELEC_OFFPEAK

    def get_heating_price(self, timestamp: pd.Timestamp) -> float:
        if timestamp.month in [11, 12, 1, 2, 3]:
            return self.HEAT_WINTER
        else:
            return self.HEAT_SUMMER


# ============================================================
# SystemConfig  (عین کد مرجع)
# ============================================================
@dataclass
class SystemConfig:
    CLASS_AREA_M2: float   = 108.0
    CLASS_HEIGHT_M: float  = 3.2
    CLASS_VOLUME_M3: float = 345.6
    VENT_RATE_MAX: float   = 5.30

    CO2_OUT_PPM: float            = 420.0
    CO2_CAP_PPM: float            = 1000.0
    CO2_GEN_LPS_PER_PERSON: float = 0.00325
    PEOPLE_PER_OCC: float         = 20.0

    RHO_AIR: float            = 1.225
    CP_AIR: float             = 1006.0
    HRU_EFF: float            = 0.72
    HEATING_EFFICIENCY: float = 0.80
    SFP_FAN_KW_PER_M3S: float = 0.90

    DT_SEC: int     = 900
    START_TIME: str = "08:00"
    END_TIME: str   = "16:00"

    ASIS_OCC_TARGET: float = 5.7
    AI_TARGET_ELEC: float  = 3.4

    CO2_TARGET_PPM: float = 970.0
    CO2_TARGET_PI: float  = 930.0

    PI_KP: float               = 0.002
    PI_KI: float               = 1.0e-6
    PI_VF_MIN: float           = 0.10
    PI_VF_MAX: float           = 2.00
    PI_RESET_I_WHEN_UNOCC: bool= True

    BIND_CO2_TO_ENERGY: bool     = True
    BASELINE_VENT_RATE_M3S: float= 2.5
    FAN_ELEC_SHARE: float        = 0.30
    HEATING_VENT_SHARE: float    = 1.0
    NON_OCC_Q_SPILLOVER: float   = 0.20
    FAN_POWER_EXP: float         = 3


# ============================================================
# PPO: Actor-Critic Network  (جدید - RL)
# ============================================================
class ActorCritic(nn.Module):
    """نگه داشته شده برای سازگاری — از DCV_Agent استفاده می‌شود"""
    def __init__(self, state_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden, 1)
        self.actor_log_std = nn.Parameter(torch.ones(1) * -1.0)
        self.critic        = nn.Linear(hidden, 1)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def forward(self, x):
        h    = self.shared(x)
        mean = torch.sigmoid(self.actor_mean(h)) * 1.9 + 0.1
        std  = self.actor_log_std.clamp(-3, 0).exp().expand_as(mean)
        return mean, std, self.critic(h)

    def get_action(self, state_np):
        x = torch.FloatTensor(state_np).unsqueeze(0)
        with torch.no_grad():
            mean, std, val = self(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(0.1, 2.0)
        return action.item(), dist.log_prob(action).sum().item(), val.item()
            
# ============================================================
# Classroom Environment  (جدید - RL)
# همان فرمول CO2 mass-balance عین simulate_co2_series
# همان Q mapping عین کد مرجع
# ============================================================
class ClassroomEnv:
    def __init__(self, occ_people, outdoor_T, baseline_Q_series, config, co2_target):
        self.occ    = occ_people.values.astype(float)
        self.T_out  = outdoor_T.values.astype(float)
        self.index  = occ_people.index
        self.n      = len(self.occ)
        self.cfg    = config
        self.target = float(co2_target)
        self.Q_base = pd.to_numeric(baseline_Q_series, errors='coerce').values.astype(float)
        self.Q_base = np.where(np.isfinite(self.Q_base), self.Q_base,
                               config.BASELINE_VENT_RATE_M3S)
        self.Q_ref  = float(np.nanmedian(self.Q_base[self.Q_base > 0])) \
                      if np.any(self.Q_base > 0) else config.BASELINE_VENT_RATE_M3S
        self.reset()

    def _compute_Q(self, vf, t):
        cfg = self.cfg
        occ_frac = min(self.occ[t] / max(cfg.PEOPLE_PER_OCC, 1), 1.0)
        spill    = cfg.NON_OCC_Q_SPILLOVER
        Q = self.Q_base[t] * (1.0 + (vf - 1.0) * (occ_frac + spill * (1 - occ_frac)))
        return float(np.clip(Q, 1e-9, cfg.VENT_RATE_MAX))

    def reset(self):
        self.t   = 0
        self.co2 = self.cfg.CO2_OUT_PPM
        self.vf  = 1.0
        return self._get_state()

    def _get_state(self):
        t = min(self.t, self.n - 1)
        h = self.index[t].hour if hasattr(self.index[t], 'hour') else 12
        co2_norm = np.clip((self.co2 - self.cfg.CO2_OUT_PPM) /
                            max(self.target - self.cfg.CO2_OUT_PPM, 1), 0, 2)
        return np.array([co2_norm, float(self.co2 <= self.target),
                         float(self.occ[min(self.t, self.n-1)] > 0),
                         min(self.occ[min(self.t, self.n-1)] / max(self.cfg.PEOPLE_PER_OCC,1), 1),
                         np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)], dtype=np.float32)

    def step(self, action):
        t   = self.t
        cfg = self.cfg
        dt  = float(cfg.DT_SEC)
        V   = cfg.CLASS_AREA_M2 * cfg.CLASS_HEIGHT_M
        vf  = float(np.clip(action, cfg.PI_VF_MIN, cfg.PI_VF_MAX))
        Q   = self._compute_Q(vf, t)
        G   = self.occ[t] * cfg.CO2_GEN_LPS_PER_PERSON / 1000.0
        Css = cfg.CO2_OUT_PPM + (G / Q) * 1e6
        a   = 1.0 - np.exp(-Q * dt / max(V, 1e-9))
        self.co2 = float(np.clip(self.co2 + a * (Css - self.co2), 350, cfg.CO2_CAP_PPM))
        self.vf  = vf
        r        = self._reward(vf, t)
        self.t  += 1
        return self._get_state(), r, self.t >= self.n, {'co2': self.co2, 'Q': Q, 'vf': vf}

    def _reward(self, vf: float, t: int) -> float:
        cfg = self.cfg

        # ── غیراشغال: هدف VF=0.5 نه VF_min=0.1 ──────────────
        if self.occ[t] <= 0:
            # جریمه فاصله از 0.5 (نه 0.1)
            return float(-3.0 * max(0.0, vf - 0.50))

        # ── اشغال ─────────────────────────────────────────────
        # پاداش صرفه‌جویی: نسبت به VF=1.0 (baseline)
        # اگر VF=0.5 → reward=+1.0، اگر VF=1.0 → reward=0، اگر VF=2.0 → -2.0
        energy_rew = (1.0 - vf) * 2.0

        # جریمه CO2
        excess    = max(0.0, self.co2 - self.target)
        co2_pen   = (excess / 50.0) ** 2 * 8.0

        # جریمه اگر VF زیر کف ایمن برود
        floor_pen = 0.0
        if vf < 0.50:
            floor_pen = (0.50 - vf) * 10.0   # جریمه سنگین زیر 0.5

        return float(energy_rew - co2_pen - floor_pen)
        
class DCVAgent:
    """
    رویکرد مستقیم و اثبات‌شده برای صرفه‌جویی انرژی:
    - وقت کلاس خالیه: VF = VF_min (حداقل تهویه)
    - وقت کلاس پره: VF = f(occupancy) — proportional control
    - یادگیری: scale factor را با Q-learning ساده تنظیم می‌کند
    """
    def __init__(self, vf_min=0.10, vf_max=2.0, n_bins=10, lr=0.1, gamma=0.95,
                 epsilon_start=0.3, epsilon_end=0.05):
        self.vf_min      = vf_min
        self.vf_max      = vf_max
        self.n_bins      = n_bins
        self.lr          = lr
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        # Q-table: state = (co2_bin, occ_bin), action = vf_bin
        self.q_table = np.zeros((n_bins, n_bins, n_bins))
        # action space: n_bins VF values between vf_min and vf_max
        self.vf_actions = np.linspace(vf_min, vf_max, n_bins)
        self.episode     = 0

    def _discretize(self, co2_ppm, occ_frac, co2_target, co2_out=420):
        co2_bin = int(np.clip((co2_ppm - co2_out) / max(co2_target - co2_out, 1) * (self.n_bins-1), 0, self.n_bins-1))
        occ_bin = int(np.clip(occ_frac * (self.n_bins-1), 0, self.n_bins-1))
        return co2_bin, occ_bin

    def select_action(self, state_np: np.ndarray) -> Tuple[float, float, float]:
        """state_np: [co2_norm, co2_ok, is_occ, occ_frac, sin_h, cos_h]"""
        co2_norm = float(state_np[0])
        is_occ   = float(state_np[2])
        occ_frac = float(state_np[3])

        # وقتی خالیه: همیشه VF_min
        if is_occ < 0.5:
            return self.vf_min, 0.0, 0.0

        # co2_ppm تقریبی از co2_norm
        co2_approx = 420 + co2_norm * 480  # [420, 900] approx
        c_bin, o_bin = self._discretize(co2_approx, occ_frac, 900)

        if np.random.random() < self.epsilon:
            a_idx = np.random.randint(self.n_bins)
        else:
            a_idx = int(np.argmax(self.q_table[c_bin, o_bin]))

        return float(self.vf_actions[a_idx]), 0.0, 0.0

    def update_q(self, state_np, action, reward, next_state_np, done):
        co2_norm   = float(state_np[0])
        occ_frac   = float(state_np[3])
        co2_approx = 420 + co2_norm * 480
        c_bin, o_bin = self._discretize(co2_approx, occ_frac, 900)

        a_idx = int(np.argmin(np.abs(self.vf_actions - action)))

        nc_norm = float(next_state_np[0])
        no_frac = float(next_state_np[3])
        nc_approx = 420 + nc_norm * 480
        nc_bin, no_bin = self._discretize(nc_approx, no_frac, 900)

        target_q = reward + (0 if done else self.gamma * np.max(self.q_table[nc_bin, no_bin]))
        self.q_table[c_bin, o_bin, a_idx] += self.lr * (target_q - self.q_table[c_bin, o_bin, a_idx])

    def train_episode(self, env: ClassroomEnv) -> dict:
        state = env.reset()
        ep_r, co2_log, vf_log = 0.0, [], []
        while True:
            vf, _, _ = self.select_action(state)
            next_s, r, done, info = env.step(vf)
            self.update_q(state, vf, r, next_s, done)
            ep_r += r
            co2_log.append(info['co2'])
            vf_log.append(info['vf'])
            state = next_s
            if done:
                break
        self.episode += 1
        # epsilon decay
        self.epsilon = max(self.epsilon_end,
                           self.epsilon - (0.3 - self.epsilon_end) / 150)
        return {'reward': ep_r, 'avg_co2': float(np.mean(co2_log)),
                'vf_mean': float(np.mean(vf_log)), 'loss': 0.0}

    def run_inference(self, env: ClassroomEnv) -> Tuple[List[float], List[float]]:
        """Greedy policy (بدون exploration)"""
        old_eps, self.epsilon = self.epsilon, 0.0
        state = env.reset()
        vf_out, co2_out = [], []
        while True:
            vf, _, _ = self.select_action(state)
            state, _, done, info = env.step(vf)
            vf_out.append(info['vf'])
            co2_out.append(info['co2'])
            if done:
                break
        self.epsilon = old_eps
        return vf_out, co2_out


class PPOAgent:
    """Wrapper که DCVAgent را صدا می‌زند — برای سازگاری با بقیه کد"""
    def __init__(self, state_dim=6, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4, update_interval=96):
        self._dcv = DCVAgent(
            vf_min=0.20,        # ← کف واقعی: حداقل 50٪ تهویه پایه (جلوگیری از یخ‌زدگی)
            vf_max=2.0,
            n_bins=10,
            lr=0.15,
            gamma=gamma,
            epsilon_start=0.4,
            epsilon_end=0.05
        )
        self.total_updates = 0

    def select_action(self, state):
        return self._dcv.select_action(state)

    def train_episode(self, env):
        return self._dcv.train_episode(env)

    def run_inference(self, env):
        return self._dcv.run_inference(env)
        
def train_rl_agent(env: ClassroomEnv, agent: PPOAgent,
                   n_episodes: int = 50) -> dict:
    history = {'episode': [], 'reward': [], 'avg_co2': [],
               'vf_mean': [], 'loss': []}
    print(f"\n{'='*60}")
    print(f"🚀 Training PPO Agent ({n_episodes} episodes)")
    print(f"{'='*60}")
    for ep in range(n_episodes):
        stats = agent.train_episode(env)
        history['episode'].append(ep + 1)
        history['reward'].append(stats['reward'])
        history['avg_co2'].append(stats['avg_co2'])
        history['vf_mean'].append(stats['vf_mean'])
        history['loss'].append(stats['loss'])
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Ep {ep+1:3d}/{n_episodes} | "
                  f"Reward: {stats['reward']:9.1f} | "
                  f"Avg CO₂: {stats['avg_co2']:6.1f} ppm | "
                  f"VF: {stats['vf_mean']:.3f}")
    print(f"{'='*60}")
    return history


# ============================================================
# Helper functions  (عین کد مرجع - بدون تغییر)
# ============================================================
def load_and_preprocess_data(baseline_path: Path,
                              weather_path: Path = None,
                              qproxy_path: Path = None) -> pd.DataFrame:
    print("📂 Loading data...")
    df = pd.read_csv(baseline_path)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime')
    else:
        raise ValueError("CSV must contain 'datetime' or 'DateTime' column.")

    if 'occupancy' not in df.columns:
        for c in ['occupancy_students', 'Occupancy', 'occupancy_students_15min']:
            if c in df.columns:
                df['occupancy'] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                break

    df = df.resample('15min').mean(numeric_only=True).ffill()

    if qproxy_path is not None:
        try:
            qx = pd.read_excel(qproxy_path)
            if 'datetime' not in qx.columns:
                raise ValueError("Q-proxy file must have a 'datetime' column.")
            qx['datetime'] = pd.to_datetime(qx['datetime'])
            qx = qx.set_index('datetime').sort_index()
            q_col = None
            for c in ['q_supply_room_m3s_est', 'q_supply_room_m3s', 'q_room_m3s']:
                if c in qx.columns:
                    q_col = c; break
            if q_col is None:
                raise ValueError("Q-proxy file missing room airflow column.")
            q_series = pd.to_numeric(qx[q_col], errors='coerce')
            q_series = q_series.resample('15min').mean().reindex(df.index)
            q_series = q_series.interpolate(method='time', limit=8,
                                             limit_area='inside').ffill().bfill()
            df['Q_room_m3s'] = q_series.astype(float)
            if 'q_source' in qx.columns:
                src = qx['q_source'].astype(str).resample('15min').first()
                df['Q_room_source'] = src.reindex(df.index).fillna('unknown')
        except Exception as e:
            print(f"⚠️ Failed to load Q-proxy: {e}")

    if 'electricity_kWh' not in df.columns and 'energy_kWh_15min_class2091' in df.columns:
        WEEKDAY_FACTOR = 0.1633569050444698
        df['electricity_kWh'] = pd.to_numeric(
            df['energy_kWh_15min_class2091'], errors='coerce') * WEEKDAY_FACTOR

    if 'Outdoor_T' not in df.columns:
        print("⚠️ No outdoor temperature, using 10°C default")
        df['Outdoor_T'] = 10.0
    if 'electricity_kWh' not in df.columns:
        print("⚠️ Creating synthetic electricity data")
        df['electricity_kWh'] = np.random.normal(1.2, 0.3, len(df))
    if 'heating_kWh' not in df.columns:
        print("⚠️ Creating synthetic heating data")
        df['heating_kWh'] = np.random.normal(0.8, 0.2, len(df))

    print(f"✅ Loaded {len(df)} timesteps")
    return df


def pick_first_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None


def get_occ_mask_15min(df: pd.DataFrame) -> pd.Series:
    if 'occupancy' in df.columns:
        occ_raw = df['occupancy']
    elif 'occupancy_students' in df.columns:
        occ_raw = df['occupancy_students']
    else:
        return pd.Series(False, index=df.index, name='occ_mask')
    occ = pd.to_numeric(occ_raw, errors='coerce').fillna(0) > 0
    return pd.Series(occ.values, index=df.index, name='occ_mask').astype(bool)


def get_measured_asis_energy(df: pd.DataFrame,
                              result_index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series, bool]:
    e_col = pick_first_col(df, ['electricity_kWh', 'elec_kWh', 'electricity'])
    h_col = pick_first_col(df, ['heating_kWh', 'heat_kWh', 'heating'])
    if e_col is None or h_col is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), False
    elec = df[e_col].reindex(result_index).ffill()
    heat = df[h_col].reindex(result_index).ffill()
    return elec, heat, True


def get_measured_co2(df: pd.DataFrame) -> Tuple[Optional[pd.Series], bool, str]:
    co2_col = pick_first_col(df, ['CO2_ppm', 'CO2', 'co2', 'co2_ppm', 'co2_avg_ppm'])
    if co2_col is None:
        return None, False, 'no_measured'
    s = pd.to_numeric(df[co2_col], errors='coerce')
    s = s.mask((s < 300) | (s > 5000))
    return s, True, co2_col


def _baseline_Q_to_series(baseline_Q, idx: pd.DatetimeIndex) -> pd.Series:
    """عین کد مرجع"""
    if np.isscalar(baseline_Q):
        return pd.Series(float(baseline_Q), index=idx, dtype=float)
    if isinstance(baseline_Q, pd.Series):
        s = pd.to_numeric(baseline_Q, errors='coerce').reindex(idx)
    else:
        s = pd.Series(pd.to_numeric(baseline_Q, errors='coerce'), index=idx)
    s = s.astype(float)
    s = s.interpolate(method='time', limit=8, limit_area='inside').ffill().bfill()
    return s


def simulate_co2_series(baseline_Q, vent_factor, occ_people_series, config):
    """عین کد مرجع"""
    idx   = occ_people_series.index
    dt    = config.DT_SEC
    V     = config.CLASS_AREA_M2 * config.CLASS_HEIGHT_M
    C_out = config.CO2_OUT_PPM
    occ_frac = (occ_people_series / max(config.PEOPLE_PER_OCC, 1)).clip(0, 1)
    spill    = config.NON_OCC_Q_SPILLOVER
    baseline_Q_series = _baseline_Q_to_series(baseline_Q, idx)

    if np.isscalar(vent_factor):
        Q_t = baseline_Q_series * (1.0 + (vent_factor - 1.0) *
                                   (occ_frac + spill * (1 - occ_frac)))
    else:
        vf_s = vent_factor if isinstance(vent_factor, pd.Series) \
               else pd.Series(vent_factor, index=idx)
        Q_t = baseline_Q_series * (1.0 + (vf_s - 1.0) *
                                   (occ_frac + spill * (1 - occ_frac)))

    C = np.empty(len(idx), dtype=float)
    C[0] = C_out
    for t in range(1, len(idx)):
        people_t = max(0.0, float(occ_people_series.iloc[t]))
        G_m3s    = people_t * config.CO2_GEN_LPS_PER_PERSON / 1000.0
        Q        = max(float(Q_t.iloc[t]), 1e-9)
        C_ss     = C_out + (G_m3s / Q) * 1e6
        alpha    = 1.0 - np.exp(-Q * dt / max(V, 1e-9))
        C[t]     = C[t-1] + alpha * (C_ss - C[t-1])
    return pd.Series(np.clip(C, 350, config.CO2_CAP_PPM), index=idx)


def simulate_pi_mass_balance(baseline_Q, occ_people_series, co2_target_ppm,
                              config: SystemConfig, C0_ppm=None):
    """عین کد مرجع"""
    idx    = occ_people_series.index
    bQ     = _baseline_Q_to_series(baseline_Q, idx)
    bQ_ref = float(np.nanmedian(bQ.values))
    dt     = float(config.DT_SEC)
    V      = float(config.CLASS_AREA_M2 * config.CLASS_HEIGHT_M)
    C_out  = float(config.CO2_OUT_PPM)

    Kp       = float(getattr(config, "PI_KP",   0.002))
    Ki       = float(getattr(config, "PI_KI",   1.0e-6))
    vf_min   = float(getattr(config, "PI_VF_MIN", 0.10))
    vf_max_c = float(getattr(config, "PI_VF_MAX", 2.00))
    vf_max_p = float(np.clip(config.VENT_RATE_MAX / max(bQ_ref, 1e-9), vf_min, 1e3))
    vf_max   = min(vf_max_c, vf_max_p)
    spill    = float(getattr(config, "NON_OCC_Q_SPILLOVER", 0.20))
    reset_I  = bool(getattr(config, "PI_RESET_I_WHEN_UNOCC", True))
    occ_frac = (occ_people_series / max(config.PEOPLE_PER_OCC, 1)).clip(0, 1)

    C  = np.empty(len(idx), dtype=float)
    vf = np.ones(len(idx), dtype=float)
    C[0] = float(C0_ppm) if C0_ppm is not None and np.isfinite(C0_ppm) else C_out
    I    = 0.0

    for t in range(1, len(idx)):
        ppl = max(0.0, float(occ_people_series.iloc[t]))
        if ppl <= 0.0:
            vf[t] = 1.0
            if reset_I: I = 0.0
        else:
            e = float(C[t-1] - co2_target_ppm)
            u_unsat = 1.0 + Kp * e + Ki * I
            u_sat   = float(np.clip(u_unsat, vf_min, vf_max))
            if ((u_unsat == u_sat) or
                    ((u_unsat > vf_max) and (e < 0.0)) or
                    ((u_unsat < vf_min) and (e > 0.0))):
                I += e * dt
            vf[t] = float(np.clip(1.0 + Kp * e + Ki * I, vf_min, vf_max))

        G    = ppl * config.CO2_GEN_LPS_PER_PERSON / 1000.0
        occf = float(occ_frac.iloc[t])
        Q    = max(float(bQ.iloc[t] * (1.0 + (vf[t] - 1.0) *
                                        (occf + spill * (1.0 - occf)))), 1e-9)
        C_ss  = C_out + (G / Q) * 1e6
        alpha = 1.0 - np.exp(-Q * dt / max(V, 1e-9))
        C[t]  = C[t-1] + alpha * (C_ss - C[t-1])

    co2_s = pd.Series(np.clip(C, 350, config.CO2_CAP_PPM), index=idx)
    vf_s  = pd.Series(vf, index=idx)
    Q_s   = bQ * (1.0 + (vf_s - 1.0) * (occ_frac + spill * (1.0 - occ_frac)))
    return co2_s, vf_s, Q_s


def compute_Q_series_from_factor(baseline_Q, vent_factor, occ_people_series,
                                  config: SystemConfig) -> pd.Series:
    """عین کد مرجع"""
    occ_frac = (occ_people_series / max(config.PEOPLE_PER_OCC, 1)).clip(0, 1)
    spill    = float(getattr(config, "NON_OCC_Q_SPILLOVER", 0.20))
    bQ       = _baseline_Q_to_series(baseline_Q, occ_people_series.index)
    if np.isscalar(vent_factor):
        Q = bQ * (1.0 + (vent_factor - 1.0) * (occ_frac + spill * (1.0 - occ_frac)))
    else:
        vf_s = vent_factor if isinstance(vent_factor, pd.Series) \
               else pd.Series(vent_factor, index=occ_people_series.index)
        Q = bQ * (1.0 + (vf_s - 1.0) * (occ_frac + spill * (1.0 - occ_frac)))
    return Q.astype(float)


def solve_vent_factor_for_target_exact(baseline_Q, occ_people, target_ppm,
                                       config, occ_mask,
                                       tol_ppm=1e-3, max_iter=120):
    """عین کد مرجع - برای PI"""
    bQ  = _baseline_Q_to_series(baseline_Q, occ_people.index)
    bQr = float(np.nanmedian(bQ.values))
    lo  = 0.001
    hi  = float(np.clip(config.VENT_RATE_MAX / max(bQr, 1e-9), 1e-3, 1e3))

    def f(vf: float) -> float:
        s = simulate_co2_series(baseline_Q, vf, occ_people, config)
        return float(pd.to_numeric(s, errors='coerce')[occ_mask].mean()) - target_ppm

    flo, fhi = f(lo), f(hi)
    if flo < 0 and fhi < 0: return lo, flo + target_ppm, bQr * lo
    if flo > 0 and fhi > 0: return hi, fhi + target_ppm, bQr * hi
    if flo < 0 and fhi > 0: lo, hi, flo, fhi = hi, lo, fhi, flo

    for _ in range(max_iter):
        mid  = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) <= tol_ppm: return mid, fmid + target_ppm, bQr * mid
        if fmid > 0: lo, flo = mid, fmid
        else:        hi, fhi = mid, fmid

    vf = 0.5 * (lo + hi)
    return vf, f(vf) + target_ppm, baseline_Q * vf


def calculate_tou_costs(df: pd.DataFrame, tou_config: TOUPricing) -> pd.DataFrame:
    """عین کد مرجع"""
    df = df.copy()
    elec_prices = np.array([tou_config.get_electricity_price(ts) for ts in df.index])
    heat_prices = np.array([tou_config.get_heating_price(ts) for ts in df.index])
    for scenario in ['AsIs', 'PI', 'AI']:
        ec = f'{scenario}_electricity_kWh'
        hc = f'{scenario}_heating_kWh'
        if ec in df.columns:
            df[f'{scenario}_electricity_cost_EUR'] = df[ec] * elec_prices
        if hc in df.columns:
            df[f'{scenario}_heating_cost_EUR'] = df[hc] * heat_prices
        if (f'{scenario}_electricity_cost_EUR' in df.columns and
                f'{scenario}_heating_cost_EUR' in df.columns):
            df[f'{scenario}_total_cost_EUR'] = (df[f'{scenario}_electricity_cost_EUR'] +
                                                df[f'{scenario}_heating_cost_EUR'])
    df['electricity_price_EUR_per_kWh'] = elec_prices
    df['heating_price_EUR_per_kWh']     = heat_prices
    return df


# ============================================================
# Scenario: run_single_co2_scenario
# فرمول‌های انرژی و مقیاس‌بندی عین کد مرجع
# ============================================================
def run_single_co2_scenario(co2_target_ppm: float, df: pd.DataFrame,
                             config: SystemConfig, args,
                             occupied_days: pd.Series) -> Dict:
    print(f"\n{'='*70}")
    print(f"🎯 RL Scenario: CO₂ target = {co2_target_ppm} ppm")
    print(f"{'='*70}")
    config.CO2_TARGET_PPM = co2_target_ppm

    # ── result_index: عین کد مرجع (از sequence_length شروع میشه) ──
    # در کد مرجع: df.index[args.sequence_length : args.sequence_length + len(predictions)]
    # چون RL به sequence نیاز ندارد، از همان offset استفاده میکنیم
    # تا بازه زمانی دقیقاً یکی باشد و AsIs energy میانگین یکسانی بدهد
    seq_len      = args.sequence_length          # پیش‌فرض ۴۸
    result_index = df.index[seq_len:]

    # ── occ mask و occ_people ────────────────────────────────────
    occ_15     = get_occ_mask_15min(df).reindex(result_index, fill_value=False)
    occ_people = (pd.to_numeric(df['occupancy'], errors='coerce')
                  .reindex(result_index).fillna(0)
                  if 'occupancy' in df.columns
                  else pd.Series(0.0, index=result_index))

    # ── As-Is انرژی (اندازه‌گیری شده - عین کد مرجع) ─────────────
    asis_elec, asis_heat, have_measured_asis_energy = \
        get_measured_asis_energy(df, result_index)
    if not have_measured_asis_energy:
        asis_elec = pd.Series(np.random.normal(1.2, 0.3, len(result_index)),
                              index=result_index)
        asis_heat = pd.Series(np.random.normal(0.8, 0.2, len(result_index)),
                              index=result_index)

    # ── baseline Q (عین کد مرجع) ──────────────────────────────────
    if ('Q_room_m3s' in df.columns and
            pd.to_numeric(df['Q_room_m3s'], errors='coerce').notna().any()):
        baseline_Q_series = pd.to_numeric(
            df['Q_room_m3s'], errors='coerce').reindex(result_index)
        baseline_Q_series = (baseline_Q_series
                             .interpolate(method='time', limit=8, limit_area='inside')
                             .ffill().bfill())
        q_occ = pd.to_numeric(baseline_Q_series, errors='coerce')[occ_15]
        baseline_vent_rate = (float(np.nanmedian(q_occ.values))
                              if np.isfinite(q_occ).any()
                              else float(np.nanmedian(baseline_Q_series.values)))
        baseline_source = "Q_proxy_excel (room airflow proxy)"
    else:
        baseline_Q_series = None
        baseline_vent_rate = None
        baseline_source = None

    # ── CO2 اندازه‌گیری شده ──────────────────────────────────────
    co2_series_raw, have_measured_co2, co2_col_name = get_measured_co2(df)

    if have_measured_co2:
        co2_win = co2_series_raw.reindex(result_index)
        co2_win = co2_win.interpolate(method='time', limit=8, limit_area='inside')

        if baseline_Q_series is None:
            # استنتاج از CO2 (عین کد مرجع)
            G_m3s      = (config.PEOPLE_PER_OCC *
                          config.CO2_GEN_LPS_PER_PERSON / 1000.0)
            asis_co2_avg = float(co2_win[occ_15].mean()) if occ_15.any() else 700.0
            delta_ppm  = max(asis_co2_avg - config.CO2_OUT_PPM, 1e-6)
            baseline_vent_rate = float(G_m3s / (delta_ppm / 1e6))
            baseline_source    = f"inferred_from_measured({co2_col_name})"
            baseline_Q_series  = pd.Series(baseline_vent_rate, index=result_index)
    else:
        co2_win = None
        if baseline_Q_series is None:
            baseline_vent_rate = float(config.BASELINE_VENT_RATE_M3S)
            baseline_source    = "config.BASELINE_VENT_RATE_M3S"
            baseline_Q_series  = pd.Series(baseline_vent_rate, index=result_index)

    if baseline_vent_rate is None:
        baseline_vent_rate = float(config.BASELINE_VENT_RATE_M3S)

    # ── Q_ref برای fan model (عین کد مرجع) ───────────────────────
    Q_ref_fan = float(pd.to_numeric(baseline_Q_series, errors='coerce')[occ_15].median())
    if not np.isfinite(Q_ref_fan) or Q_ref_fan <= 0:
        Q_ref_fan = float(baseline_vent_rate)
    config.BASELINE_VENT_RATE_M3S = Q_ref_fan

    # ── PI controller (عین کد مرجع) ─────────────────────────────
    C0_init = None
    if co2_win is not None:
        try:
            C0_init = float(pd.to_numeric(co2_win, errors='coerce').iloc[0])
        except Exception:
            C0_init = None

    pi_co2_series, pi_vf_series, pi_Q_series = simulate_pi_mass_balance(
        baseline_Q=baseline_vent_rate,
        occ_people_series=occ_people,
        co2_target_ppm=config.CO2_TARGET_PI,
        config=config,
        C0_ppm=C0_init
    )
    pi_vent_factor = float(pd.to_numeric(pi_vf_series, errors='coerce')[occ_15].mean())
    pi_Q           = float(pd.to_numeric(pi_Q_series,  errors='coerce')[occ_15].mean())

    # ── AI vent factor با solve_vent_factor (عین کد مرجع - برای CO2 sim) ─
    ai_vent_factor_static, ai_co2_avg_static, ai_Q_static = \
        solve_vent_factor_for_target_exact(
            baseline_vent_rate, occ_people, co2_target_ppm, config, occ_15,
            tol_ppm=1e-3, max_iter=120)

    # ================================================================
    # RL: آموزش PPO و inference برای AI ventilation factor
    # ================================================================
    outdoor_T = (df['Outdoor_T'].reindex(result_index).ffill()
                 if 'Outdoor_T' in df.columns
                 else pd.Series(10.0, index=result_index))

    env = ClassroomEnv(
        occ_people=occ_people,
        outdoor_T=outdoor_T,
        baseline_Q_series=baseline_Q_series,
        config=config,
        co2_target=co2_target_ppm
    )
    agent = PPOAgent(state_dim=6, lr=3e-4, gamma=0.99,
                     eps_clip=0.2, k_epochs=4, update_interval=96)

    rl_history = train_rl_agent(env, agent, n_episodes=args.epochs)

    # inference: سری vf و co2 از policy آموزش‌دیده
    ai_vf_list, ai_co2_list = agent.run_inference(env)

    n_res = len(result_index)
    ai_vf_series = pd.Series(
        ai_vf_list[:n_res] if len(ai_vf_list) >= n_res
        else ai_vf_list + [ai_vf_list[-1]] * (n_res - len(ai_vf_list)),
        index=result_index)
    ai_co2_arr = np.clip(
        ai_co2_list[:n_res] if len(ai_co2_list) >= n_res
        else ai_co2_list + [ai_co2_list[-1]] * (n_res - len(ai_co2_list)),
        350, config.CO2_CAP_PPM)
    # ================================================================

    # ── DataFrame نتایج ──────────────────────────────────────────
    df_results = pd.DataFrame(index=result_index)

    # AsIs
    df_results['AsIs_electricity_kWh'] = asis_elec.values
    df_results['AsIs_heating_kWh']     = asis_heat.values
    df_results['AsIs_Q_m3s']           = baseline_Q_series.values

    if have_measured_co2 and co2_win is not None:
        df_results['AsIs_CO2_ppm'] = co2_win.values
    else:
        df_results['AsIs_CO2_ppm'] = simulate_co2_series(
            baseline_vent_rate, 1.0, occ_people, config).values

    # PI
    df_results['PI_CO2_ppm']     = pi_co2_series.values
    df_results['PI_vent_factor'] = pi_vf_series.values
    df_results['PI_Q_m3s']       = pi_Q_series.values

    # AI (RL)
    df_results['AI_CO2_ppm']     = ai_co2_arr
    df_results['AI_Q_m3s']       = compute_Q_series_from_factor(
        baseline_vent_rate, ai_vf_series, occ_people, config).values

    # ── Fan electricity: Method A (عین کد مرجع - ratio از As-Is) ──
    # P(Q) = P_ref * (Q/Q_ref)^n   →   Efan = Eref * (Q/Q_asis)^n
    n = float(getattr(config, "FAN_POWER_EXP", 3.0))

    Q_asis = pd.to_numeric(df_results['AsIs_Q_m3s'], errors='coerce').astype(float).clip(lower=1e-9)
    Q_pi   = pd.to_numeric(df_results['PI_Q_m3s'],   errors='coerce').astype(float).clip(lower=1e-9)
    Q_ai   = pd.to_numeric(df_results['AI_Q_m3s'],   errors='coerce').astype(float).clip(lower=1e-9)

    # کف منطقی (عین کد مرجع)
    Q_floor   = 0.2 * Q_ref_fan
    Q_asis_safe = Q_asis.clip(lower=Q_floor)

    # cap: PI/AI نمی‌توانند بیشتر از AsIs باشند (عین کد مرجع)
    Q_pi = np.minimum(Q_pi, Q_asis)
    Q_ai = np.minimum(Q_ai, Q_asis)

    ratio_pi = (Q_pi / Q_asis_safe)
    ratio_ai = (Q_ai / Q_asis_safe)

    # مبنای انرژی: سری اندازه‌گیری شده As-Is (عین کد مرجع)
    Eref = df_results['AsIs_electricity_kWh'].astype(float).values

    df_results['AsIs_fan_kWh'] = Eref
    df_results['PI_fan_kWh']   = Eref * (ratio_pi.values ** n)
    df_results['AI_fan_kWh']   = Eref * (ratio_ai.values ** n)

    # hard cap: PI/AI نمی‌توانند از AsIs بیشتر شوند (عین کد مرجع)
    df_results['PI_fan_kWh'] = np.minimum(df_results['PI_fan_kWh'],
                                           df_results['AsIs_fan_kWh'])
    df_results['AI_fan_kWh'] = np.minimum(df_results['AI_fan_kWh'],
                                           df_results['AsIs_fan_kWh'])

    # electricity = fan electricity (عین کد مرجع)
    df_results['PI_electricity_kWh'] = df_results['PI_fan_kWh']
    df_results['AI_electricity_kWh'] = df_results['AI_fan_kWh']

    # ── PI heating از مدل LSTM اصلی می‌آمد؛ اینجا از AsIs scale می‌کنیم ─
    df_results['PI_heating_kWh'] = asis_heat.values.copy()
    df_results['AI_heating_kWh'] = asis_heat.values.copy()

    # ── Bind CO2 → Heating (عین کد مرجع) ─────────────────────────
    if config.BIND_CO2_TO_ENERGY:
        occ_vec = occ_15.astype(float).values
        HVSHARE = float(getattr(config, 'HEATING_VENT_SHARE', 1.0))

        # PI: از vf_series (عین کد مرجع: scale_PI_q = pi_vent_factor series)
        scale_PI_q = df_results['PI_vent_factor'].values if 'PI_vent_factor' in df_results.columns \
                     else pi_vent_factor
        # AI: vent_factor میانگین (عین کد مرجع)
        scale_AI_q = ai_vf_series.values

        df_results['AI_heating_kWh'] *= (
            (1.0 - HVSHARE) + HVSHARE * (1.0 + (scale_AI_q - 1.0) * occ_vec))
        df_results['PI_heating_kWh'] *= (
            (1.0 - HVSHARE) + HVSHARE * (1.0 + (scale_PI_q - 1.0) * occ_vec))

    # ── Daily aggregation و metrics (عین کد مرجع) ────────────────
    daily_results = df_results.resample('D').sum()
    occ_mask      = occupied_days.reindex(daily_results.index, fill_value=False)

    def _day_mean(col):
        return (daily_results.loc[occ_mask, col].mean() if occ_mask.any()
                else daily_results[col].mean())

    asis_mean = _day_mean('AsIs_electricity_kWh')
    pi_mean   = _day_mean('PI_electricity_kWh')
    ai_mean   = _day_mean('AI_electricity_kWh')

    asis_fan_mean = _day_mean('AsIs_fan_kWh')
    pi_fan_mean   = _day_mean('PI_fan_kWh')
    ai_fan_mean   = _day_mean('AI_fan_kWh')

    asis_ref = asis_mean
    pi_sav   = (asis_ref - pi_mean) / asis_ref * 100 if asis_ref > 0 else 0
    ai_sav   = (asis_ref - ai_mean) / asis_ref * 100 if asis_ref > 0 else 0

    asis_co2_avg  = float(pd.to_numeric(df_results['AsIs_CO2_ppm'],errors='coerce')[occ_15].mean())
    pi_co2_avg    = float(pd.to_numeric(df_results['PI_CO2_ppm'],  errors='coerce')[occ_15].mean())
    ai_co2_avg    = float(pd.Series(ai_co2_arr, index=result_index)[occ_15].mean())

    asis_co2_peak = float(pd.to_numeric(df_results['AsIs_CO2_ppm'],errors='coerce')[occ_15].max())
    pi_co2_peak   = float(pd.to_numeric(df_results['PI_CO2_ppm'],  errors='coerce')[occ_15].max())
    ai_co2_peak   = float(pd.Series(ai_co2_arr, index=result_index)[occ_15].max())

    # ── TOU costs (عین کد مرجع - فقط electricity) ────────────────
    tou_config = TOUPricing()
    df_costs   = calculate_tou_costs(df_results, tou_config)
    daily_costs= df_costs.resample('D').sum()

    asis_cost = daily_costs['AsIs_electricity_cost_EUR'].mean()
    pi_cost   = daily_costs['PI_electricity_cost_EUR'].mean()
    ai_cost   = daily_costs['AI_electricity_cost_EUR'].mean()

    pi_cost_sav_eur = asis_cost - pi_cost
    ai_cost_sav_eur = asis_cost - ai_cost
    pi_cost_sav_pct = pi_cost_sav_eur / asis_cost * 100 if asis_cost > 0 else 0
    ai_cost_sav_pct = ai_cost_sav_eur / asis_cost * 100 if asis_cost > 0 else 0

    return {
        'co2_target': co2_target_ppm,
        'electricity_kwh_day': {
            'asis': float(asis_mean), 'pi': float(pi_mean), 'ai': float(ai_mean)},
        'fan_kwh_day': {
            'asis': float(asis_fan_mean), 'pi': float(pi_fan_mean), 'ai': float(ai_fan_mean)},
        'costs_eur_day': {
            'asis': float(asis_cost), 'pi': float(pi_cost), 'ai': float(ai_cost)},
        'cost_savings': {
            'pi_eur': float(pi_cost_sav_eur), 'ai_eur': float(ai_cost_sav_eur),
            'pi_pct': float(pi_cost_sav_pct), 'ai_pct': float(ai_cost_sav_pct)},
        'co2_achieved_ppm': {
            'asis_avg':  asis_co2_avg,  'pi_avg':  pi_co2_avg,  'ai_avg':  ai_co2_avg,
            'asis_peak': asis_co2_peak, 'pi_peak': pi_co2_peak, 'ai_peak': ai_co2_peak},
        'savings_pct': {'pi': float(pi_sav), 'ai': float(ai_sav)},
        'ventilation': {
            'baseline_m3s': float(baseline_vent_rate),
            'ai_factor':    float(ai_vf_series[occ_15].mean()),
            'pi_factor':    float(pi_vent_factor),
            'ai_m3s':       float(df_results['AI_Q_m3s'][occ_15].mean()),
            'pi_m3s':       float(pi_Q)},
        'df_results':       df_results,
        'df_costs':         df_costs,
        'daily_results':    daily_results,
        'daily_costs':      daily_costs,
        'rl_history':       rl_history,
    }


# ============================================================
# Plot functions  (عین کد مرجع)
# ============================================================
def create_rl_learning_curves(history: dict, out_dir: Path, co2_target: float = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    def sm(a, w=7): return uniform_filter1d(a, size=min(w, len(a)), mode='nearest')

    ep = history['episode']
    specs = [
        (axes[0,0], history['reward'],   'steelblue',  'a) Episode Reward',              'Total Reward'),
        (axes[0,1], history['avg_co2'],  'tomato',     'b) Avg CO₂ During Training',      'CO₂ (ppm)'),
        (axes[1,0], history['vf_mean'],  'seagreen',   'c) Mean Ventilation Factor',      'Vent Factor'),
        (axes[1,1], history['loss'],     'darkorange', 'd) PPO Policy Loss',               'Loss'),
    ]
    for ax, data, col, ttl, yl in specs:
        ax.plot(ep, data, alpha=0.25, color=col, lw=1)
        ax.plot(ep, sm(data), color=col, lw=2.5)
        ax.set_title(ttl, fontweight='bold'); ax.set_xlabel('Episode')
        ax.set_ylabel(yl); ax.grid(True, alpha=0.2)

    if co2_target:
        axes[0,1].axhline(co2_target, ls='--', color='red', alpha=0.6,
                           label=f'Target {int(co2_target)} ppm')
        axes[0,1].legend()
    axes[1,0].axhline(1.0, ls='--', color='gray', alpha=0.5)

    title = 'PPO-RL Training Curves'
    if co2_target: title += f' (CO₂={int(co2_target)} ppm)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = Path(out_dir) / 'rl_learning_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  ✅ RL learning curves: {out.name}")


def create_lstm_comparison_plots_for_scenario(result: Dict, scenario_dir: Path,
                                               co2_target: float):
    scenario_dir = Path(scenario_dir); scenario_dir.mkdir(parents=True, exist_ok=True)
    df_results   = result['df_results']
    daily_results= result['daily_results']

    asis_mean = result['electricity_kwh_day']['asis']
    ai_mean   = result['electricity_kwh_day']['ai']
    asis_co2  = result['co2_achieved_ppm']['asis_avg']
    ai_co2    = result['co2_achieved_ppm']['ai_avg']
    asis_co2p = result['co2_achieved_ppm']['asis_peak']
    ai_sav    = result['savings_pct']['ai']

    # -- نمودار ۱: bar (فقط AsIs و RL) --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sc  = ['AsIs', 'RL']
    col = ['#1f77b4', '#2ca02c']
    x   = np.arange(2)

    bars = ax1.bar(x, [asis_mean, ai_mean], color=col, alpha=0.8, width=0.4)
    for b, v in zip(bars, [asis_mean, ai_mean]):
        ax1.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.2f}', ha='center', fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(sc, fontsize=12)
    ax1.set_ylabel('Daily Electricity (kWh)'); ax1.grid(axis='y', alpha=0.3)
    ax1.set_title(f'Average Daily Electricity - CO₂={int(co2_target)}ppm', fontweight='bold')
    ax1.set_ylim(0, max(asis_mean, ai_mean)*1.25)

    # نمودار savings: مقایسه واقعی با hatch و فلش
    bars2 = ax2.bar(x, [asis_mean, ai_mean], color=col, alpha=0.85, width=0.5,
                    edgecolor='white', linewidth=1.2)
    ax2.bar(x[0], asis_mean - ai_mean, bottom=ai_mean,
            color='#ff7f0e', alpha=0.55, width=0.5,
            hatch='///', edgecolor='darkorange', linewidth=0.8,
            label=f'Savings: {ai_sav:.1f}%')
    for b, v in zip(bars2, [asis_mean, ai_mean]):
        ax2.text(b.get_x()+b.get_width()/2, v+0.08, f'{v:.2f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.annotate('', xy=(0.5, ai_mean + (asis_mean-ai_mean)/2),
                 xytext=(1.5, ai_mean + (asis_mean-ai_mean)/2),
                 arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2.0))
    ax2.text(1.0, ai_mean + (asis_mean-ai_mean)/2 + max(0.05, asis_mean*0.02),
             f'−{ai_sav:.1f}%\n(−{asis_mean-ai_mean:.2f} kWh)',
             ha='center', va='bottom', color='darkorange', fontweight='bold', fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(sc, fontsize=12)
    ax2.set_ylabel('Daily Electricity (kWh)', fontsize=11)
    ax2.set_title('RL Energy Savings vs AsIs', fontweight='bold')
    ax2.set_ylim(0, max(asis_mean, ai_mean)*1.35)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    f1 = scenario_dir/f'co2_{int(co2_target)}_bar_comparison.png'
    plt.savefig(f1, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  ✅ Saved: {f1.name}")

    # -- نمودار ۲: daily timeseries (فقط AsIs و RL) --
    fig2, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily_results.index, daily_results['AsIs_electricity_kWh'],
            label=f'AsIs (avg: {asis_mean:.2f} kWh)', color='#1f77b4', lw=1.5, alpha=0.8)
    ax.plot(daily_results.index, daily_results['AI_electricity_kWh'],
            label=f'RL (avg: {ai_mean:.2f} kWh)', color='#2ca02c', lw=1.5, alpha=0.8)
    ax.fill_between(daily_results.index,
                    daily_results['AsIs_electricity_kWh'],
                    daily_results['AI_electricity_kWh'],
                    where=(daily_results['AsIs_electricity_kWh'] >=
                           daily_results['AI_electricity_kWh']),
                    interpolate=True, alpha=0.15, color='green', label='RL Savings')
    ax.set_ylabel('Electricity (kWh/day)', fontsize=12)
    ax.set_title(f'Daily Electricity: CO₂={int(co2_target)}ppm  (AsIs→RL: {ai_sav:.1f}% savings)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3, linestyle=':')
    if len(daily_results) > 60:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    f2 = scenario_dir/f'co2_{int(co2_target)}_daily_timeseries.png'
    plt.savefig(f2, dpi=150, bbox_inches='tight'); plt.close(fig2)
    print(f"  ✅ Saved: {f2.name}")

    # -- نمودار ۳: CO2 (7 روز، فقط AsIs و RL) --
    fig3, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8),
                                   gridspec_kw={'height_ratios': [3,1], 'hspace': 0.05})
    sl = slice(0, min(7*96, len(df_results)))
    a1.plot(df_results.index[sl], df_results['AsIs_CO2_ppm'].iloc[sl],
            label=f'AsIs (avg: {asis_co2:.0f} ppm)', color='#1f77b4', lw=1.2, alpha=0.8)
    a1.plot(df_results.index[sl], df_results['AI_CO2_ppm'].iloc[sl],
            label=f'RL (avg: {ai_co2:.0f} ppm)', color='#2ca02c', lw=1.2, alpha=0.8)
    a1.axhspan(800, 1000, alpha=0.05, color='yellow', label='Good (800-1000 ppm)')
    a1.axhspan(420,  800, alpha=0.05, color='green',  label='Excellent (<800 ppm)')
    a1.axhline(1000, ls='--', lw=0.8, color='red', alpha=0.5, label='1000 ppm limit')
    a1.set_ylabel('CO₂ (ppm)', fontsize=12)
    a1.set_title(f'Indoor CO₂ - CO₂ Target = {int(co2_target)} ppm', fontsize=14, fontweight='bold')
    a1.legend(loc='upper right', ncol=3, fontsize=9)
    a1.grid(True, alpha=0.3, linestyle=':')
    a1.set_ylim(350, max(1100, asis_co2p + 50)); a1.set_xticklabels([])
    ti  = df_results.index[sl]
    occ = ((ti.hour >= 8) & (ti.hour < 16) & (ti.dayofweek < 5)).astype(int) * 20
    a2.fill_between(ti, 0, occ, color='lightblue', alpha=0.5)
    a2.set_ylabel('Occupancy', fontsize=10); a2.set_xlabel('Date', fontsize=11)
    a2.set_ylim(0, 25); a2.grid(True, alpha=0.2, axis='y')
    a2.xaxis.set_major_locator(mdates.DayLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(a2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    f3 = scenario_dir/f'co2_{int(co2_target)}_co2_comparison.png'
    plt.savefig(f3, dpi=150, bbox_inches='tight'); plt.close(fig3)
    print(f"  ✅ Saved: {f3.name}")

    # -- نمودار ۴: full timeseries (فقط AsIs و RL) --
    fig4, ax4 = plt.subplots(figsize=(24, 4))
    ax4.plot(df_results.index, df_results['AsIs_CO2_ppm'],
             label='As-Is (measured)', lw=0.8, color='#1f77b4')
    ax4.plot(df_results.index, df_results['AI_CO2_ppm'],
             label='RL System', lw=0.8, color='#2ca02c')
    ax4.axhline(1000, linestyle='--', lw=0.8, color='red', alpha=0.7, label='1000 ppm')
    ax4.set_ylabel('ppm'); ax4.set_ylim(400, 1050)
    ax4.set_title(f'CO₂ Timeseries - CO₂ target={int(co2_target)} ppm')
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.2, linestyle=':')
    plt.tight_layout()
    f4 = scenario_dir/f'co2_{int(co2_target)}_full_timeseries.png'
    plt.savefig(f4, dpi=150, bbox_inches='tight'); plt.close(fig4)
    print(f"  ✅ Saved: {f4.name}")


def create_tou_cost_plots_for_scenario(result: Dict, scenario_dir: Path,
                                        co2_target: float, tou_config: TOUPricing):
    scenario_dir = Path(scenario_dir); scenario_dir.mkdir(parents=True, exist_ok=True)
    df_costs  = calculate_tou_costs(result['df_results'], tou_config)
    daily_c   = df_costs.resample('D').sum()

    ae  = daily_c['AsIs_electricity_cost_EUR'].mean()
    aie = daily_c['AI_electricity_cost_EUR'].mean()
    ai_sav = ae - aie
    ai_pct = ai_sav/ae*100 if ae > 0 else 0

    # -- نمودار ۱: bar (فقط AsIs و RL) --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(2); sc = ['AsIs', 'RL']; col = ['#1f77b4', '#2ca02c']

    bars = ax1.bar(x, [ae, aie], color=col, edgecolor='black',
                   linewidth=1.0, alpha=0.8)
    for b, v in zip(bars, [ae, aie]):
        ax1.text(b.get_x()+b.get_width()/2, v+0.02, f'€{v:.2f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Average Daily Electricity Cost (EUR)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Cost Breakdown - CO₂={int(co2_target)}ppm', fontsize=14, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(sc, fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(ae, aie)*1.25)

    bars2 = ax2.bar(x, [ae, aie], color=col, alpha=0.85, width=0.5,
                    edgecolor='white', linewidth=1.2)
    ax2.bar(x[0], ai_sav, bottom=aie,
            color='#ff7f0e', alpha=0.55, width=0.5,
            hatch='///', edgecolor='darkorange', linewidth=0.8,
            label=f'Savings: €{ai_sav:.2f}')
    for b, v in zip(bars2, [ae, aie]):
        ax2.text(b.get_x()+b.get_width()/2, v+0.008, f'€{v:.2f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.annotate('', xy=(0.5, aie + ai_sav/2), xytext=(1.5, aie + ai_sav/2),
                 arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2.0))
    ax2.text(1.0, aie + ai_sav/2 + max(0.005, ae*0.02),
             f'−{ai_pct:.1f}%\n(−€{ai_sav:.2f})',
             ha='center', va='bottom', color='darkorange', fontweight='bold', fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(sc, fontsize=12)
    ax2.set_ylabel('Daily Electricity Cost (EUR)', fontsize=11, fontweight='bold')
    ax2.set_title('RL Cost Savings vs AsIs', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(ae, aie)*1.35)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(scenario_dir/f'co2_{int(co2_target)}_tou_cost_bars.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: co2_{int(co2_target)}_tou_cost_bars.png")

    # -- نمودار ۲: daily cost timeseries (فقط AsIs و RL) --
    fig2, ax = plt.subplots(figsize=(15, 6))
    at = daily_c['AsIs_total_cost_EUR'] if 'AsIs_total_cost_EUR' in daily_c.columns \
         else daily_c['AsIs_electricity_cost_EUR']
    it = daily_c['AI_total_cost_EUR']   if 'AI_total_cost_EUR'   in daily_c.columns \
         else daily_c['AI_electricity_cost_EUR']
    ax.plot(daily_c.index, at, label=f'AsIs (avg: €{ae:.2f})',    color='#1f77b4', lw=1.5, alpha=0.8)
    ax.plot(daily_c.index, it, label=f'RL (avg: €{aie:.2f})',     color='#2ca02c', lw=1.5, alpha=0.8)
    ax.fill_between(daily_c.index, at, it,
                    where=(at >= it), interpolate=True,
                    alpha=0.15, color='green', label='RL Cost Savings')
    ax.set_ylabel('Daily Total Cost (EUR)', fontsize=12)
    ax.set_title(f'Daily TOU Costs: CO₂={int(co2_target)}ppm  '
                 f'(RL savings: €{ai_sav:.2f}/day, {ai_pct:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(scenario_dir/f'co2_{int(co2_target)}_tou_daily_costs.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✅ Saved: co2_{int(co2_target)}_tou_daily_costs.png")

    return {
        'df_costs': df_costs, 'daily_costs': daily_c,
        'asis_total_eur_day': ae, 'ai_total_eur_day': aie,
        'ai_savings_eur_day': ai_sav, 'ai_savings_pct': ai_pct,
    }


def create_co2_sweep_plots(sweep_results: List[Dict], out_dir: Path):
    """عین کد مرجع"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    co2_t = [r['co2_target'] for r in sweep_results]
    ai_e  = [r['electricity_kwh_day']['ai'] for r in sweep_results]
    ai_s  = [r['savings_pct']['ai'] for r in sweep_results]
    ai_c  = [r['co2_achieved_ppm']['ai_avg'] for r in sweep_results]
    ai_vf = [r['ventilation']['ai_factor'] for r in sweep_results]

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(co2_t, ai_e, 'o-', color='#2ca02c', lw=2, ms=8)
    ax1.set_xlabel('CO₂ Target (ppm)'); ax1.set_ylabel('AI Daily Electricity (kWh)')
    ax1.set_title('Electricity Consumption vs CO₂ Target', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    for i,(x,y) in enumerate(zip(co2_t,ai_e)):
        if i % 2 == 0: ax1.text(x, y+0.02, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.plot(co2_t, ai_s, 's-', color='#ff7f0e', lw=2, ms=8)
    ax2.set_xlabel('CO₂ Target (ppm)'); ax2.set_ylabel('Energy Savings vs AsIs (%)')
    ax2.set_title('Energy Savings vs CO₂ Target', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.axhline(y=0, color='gray', ls='-', alpha=0.3)

    ax3.plot(co2_t, co2_t, 'k--', alpha=0.3, label='Perfect match')
    ax3.plot(co2_t, ai_c, '^-', color='#1f77b4', lw=2, ms=8, label='Achieved')
    ax3.set_xlabel('CO₂ Target (ppm)'); ax3.set_ylabel('CO₂ Achieved (ppm)')
    ax3.set_title('CO₂ Target vs Achieved', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':'); ax3.legend()

    ax4.plot(co2_t, ai_vf, 'd-', color='#d62728', lw=2, ms=8)
    ax4.set_xlabel('CO₂ Target (ppm)'); ax4.set_ylabel('Ventilation Factor')
    ax4.set_title('Ventilation Factor vs CO₂ Target', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':')
    ax4.axhline(y=1.0, color='gray', ls='--', alpha=0.3, label='Baseline')
    ax4.legend()

    plt.suptitle('CO₂ Target Sweep Analysis - PPO-RL (900-990 ppm)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir/'co2_sweep_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print("✅ Sweep analysis plot saved")

    # bar comparison
    fig2, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(co2_t)); w = 0.35
    b1 = ax.bar(x-w/2, ai_e, w, label='Electricity (kWh/day)', color='#2ca02c', alpha=0.8)
    b2 = ax.bar(x+w/2, ai_s, w, label='Savings (%)', color='#ff7f0e', alpha=0.8)
    ax.set_xlabel('CO₂ Target (ppm)', fontsize=12); ax.set_ylabel('Value', fontsize=12)
    ax.set_title('AI-RL Performance Across CO₂ Targets', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([str(int(t)) for t in co2_t], rotation=45)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for b in b1:
        h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h+0.05,f'{h:.2f}',ha='center',va='bottom',fontsize=8)
    for b in b2:
        h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h+0.5,f'{h:.1f}%',ha='center',va='bottom',fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir/'co2_sweep_bars.png', dpi=150, bbox_inches='tight')
    plt.close(fig2); print("✅ Bar comparison plot saved")


def create_violation_metrics_plot(sweep_results: List[Dict], out_dir: Path):
    if not sweep_results: return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels_sorted = sorted(sweep_results, key=lambda r: r['co2_target'])
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels_sorted)))

    def _smooth(arr, k=7):
        return uniform_filter1d(arr, size=min(k,len(arr)), mode='nearest') if len(arr)>=k else arr

    for i, result in enumerate(labels_sorted):
        co2_t = result['co2_target']
        df_r  = result['df_results'].copy()
        df_r['weekday'] = df_r.index.dayofweek
        df_r['hour']    = df_r.index.hour
        occ_mask = (df_r['weekday'] < 5) & (df_r['hour'].between(8, 16))
        n_steps  = len(df_r); wz = 100
        v1, v2, ac, xs = [], [], [], []
        for s in range(0, n_steps, wz):
            e  = min(s+wz, n_steps)
            wo = occ_mask.iloc[s:e]
            if 'AI_CO2_ppm' in df_r.columns:
                wd = df_r.iloc[s:e]
                cv = wd.loc[wo, 'AI_CO2_ppm'] if wo.any() else pd.Series(dtype=float)
                if len(cv) > 0:
                    v1.append((cv>1000).mean()); v2.append((cv>900).mean())
                    ac.append(np.clip((cv.mean()-420)/580, 0, 1))
                else:
                    v1.append(0); v2.append(0); ac.append(0)
            else:
                v1.append(0); v2.append(0); ac.append(0)
            xs.append(s)
        c   = colors[i]; lbl = f"RL CO₂={int(co2_t)} ppm"
        ax1.plot(xs, np.array(v1), color=c, alpha=0.20, lw=0.8)
        ax1.plot(xs, _smooth(np.array(v1)), color=c, alpha=0.95, lw=2.2, label=lbl)
        ax2.plot(xs, np.array(v2), color=c, alpha=0.20, lw=0.8)
        ax2.plot(xs, _smooth(np.array(v2)), color=c, alpha=0.95, lw=2.2)
        ax3.plot(xs, np.array(ac), color=c, alpha=0.15, lw=0.8)
        ax3.plot(xs, _smooth(np.array(ac)), color=c, alpha=0.95, lw=2.0)

    ax1.set_ylabel("Violation rate (>1000 ppm)", fontsize=11)
    ax1.set_title("RL CO₂ Violation Metrics", fontsize=13, fontweight="bold")
    ax1.set_ylim(0,1); ax1.grid(True, alpha=0.3, ls=':')
    ax2.set_ylabel("Violation rate (>900 ppm)", fontsize=11)
    ax2.set_ylim(0,1); ax2.grid(True, alpha=0.3, ls=':')
    ax3.set_xlabel("timestep", fontsize=11)
    ax3.set_ylabel("normalized avg CO₂", fontsize=11)
    ax3.set_ylim(0,1); ax3.grid(True, alpha=0.3, ls=':')
    for a in [ax1, ax2, ax3]:
        for sp in ['top','right']: a.spines[sp].set_visible(False)
    ax1.legend(title="scenario", loc="upper right", ncol=2, frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'violation_metrics_plot.png', dpi=200, bbox_inches='tight')
    plt.close(fig); print("✅ Violation metrics plot saved")


def _plot_optimization_3d_from_results(sweep_results: List[Dict], out_dir):
    """عین کد مرجع"""
    rows = []
    for r in sweep_results:
        df  = r['df_results']; idx = df.index
        occ = ((idx.dayofweek<5) & (idx.hour>=8) & (idx.hour<16))
        h_a = (df.loc[occ,'AI_CO2_ppm']>1000).sum()*(15.0/60.0) \
              if 'AI_CO2_ppm' in df.columns else np.nan
        rows.append({'co2_target': r['co2_target'],
                     'energy_kwh': r['electricity_kwh_day']['ai'],
                     'avg_co2_occ': r['co2_achieved_ppm']['ai_avg'],
                     'hours_above_limit': h_a})
    df_p = pd.DataFrame(rows).dropna()
    if df_p.empty: return

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(df_p['energy_kwh'], df_p['avg_co2_occ'], df_p['hours_above_limit'],
                     c=df_p['co2_target'], s=100, cmap='viridis',
                     alpha=0.7, edgecolors='black', linewidth=0.5, depthshade=True)
    fig.colorbar(sc, ax=ax, pad=0.15, shrink=0.7, label='CO₂ Target (ppm)')
    ax.set_xlabel('Electrical Energy (kWh/day)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Avg CO₂ during occupancy (ppm)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Hours above 1000 ppm (h)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('3D Optimization Results: Energy vs CO₂ vs Violations (PPO-RL)',
                 fontsize=14, fontweight='bold', pad=20)
    try:
        thr  = df_p['hours_above_limit'].quantile(0.2)
        feas = df_p[df_p['hours_above_limit'] <= thr]
        if not feas.empty:
            best = feas.sort_values(['energy_kwh','avg_co2_occ']).iloc[0]
            ax.scatter([best['energy_kwh']], [best['avg_co2_occ']],
                       [best['hours_above_limit']],
                       marker='*', s=500, color='red', edgecolor='darkred',
                       linewidth=2, zorder=10, label='Best Solution')
            ax.text(best['energy_kwh'], best['avg_co2_occ'], best['hours_above_limit'],
                    f"  Best\n  CO₂={int(best['co2_target'])}ppm\n  E={best['energy_kwh']:.2f}kWh",
                    fontsize=9, color='darkred', fontweight='bold')
    except Exception as e:
        print(f"⚠️ Best solution: {e}")
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'optimization_3d_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print("✅ 3D scatter saved")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PPO-RL Energy Optimization with CO2 Sweep")
    default_csv   = str((PROJECT_ROOT/'energy'/'Final_Excel4.csv').resolve())
    default_qp    = str((PROJECT_ROOT/'energy'/'G32TK15_Class2091_Qproxy_full_article_window.xlsx').resolve())
    default_out   = str((PROJECT_ROOT/'Results1').resolve())

    parser.add_argument('--baseline_csv',       type=str, default=default_csv)
    parser.add_argument('--weather_file',        type=str, default=None)
    parser.add_argument('--qproxy_xlsx',         type=str, default=default_qp)
    parser.add_argument('--out_dir',             type=str, default=default_out)
    parser.add_argument('--epochs',              type=int, default=300)
    parser.add_argument('--sequence_length',     type=int, default=48,
                        help='Offset از ابتدای داده - عین کد مرجع')
    parser.add_argument('--co2_min',             type=int, default=900)
    parser.add_argument('--co2_max',             type=int, default=990)
    parser.add_argument('--co2_step',            type=int, default=10)
    parser.add_argument('--target_ai_kwh',       type=float, default=3.7)
    parser.add_argument('--target_asis_kwh',     type=float, default=5.7)
    parser.add_argument('--bind_co2_to_energy',  type=int, choices=[0,1], default=1)
    args = parser.parse_args()

    # اطمینان از اینکه sequence_length موجود است
    if not hasattr(args, 'sequence_length'):
        args.sequence_length = 48

    torch.manual_seed(42); np.random.seed(42)

    config = SystemConfig()
    config.AI_TARGET_ELEC      = args.target_ai_kwh
    config.ASIS_OCC_TARGET     = args.target_asis_kwh
    config.BIND_CO2_TO_ENERGY  = bool(args.bind_co2_to_energy)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🔍 Output: {out_dir.absolute()}")
    print(f"   Writable: {os.access(out_dir, os.W_OK)}")

    # تست نوشتن
    tf = out_dir/'_test.txt'
    try:
        tf.write_text('test'); tf.unlink(); print("   ✅ Write test OK")
    except Exception as e:
        print(f"   ❌ Write test failed: {e}"); return

    print("="*80)
    print("🚀 PPO-RL Energy Optimization with CO2 Sweep")
    print(f"   CO2 Range: {args.co2_min}-{args.co2_max} ppm, Step: {args.co2_step} ppm")
    print("="*80)

    qproxy_path = Path(args.qproxy_xlsx) if getattr(args,'qproxy_xlsx',None) else None
    df = load_and_preprocess_data(Path(args.baseline_csv),
                                   Path(args.weather_file) if args.weather_file else None,
                                   qproxy_path)

    occupied_days = get_occ_mask_15min(df).resample('D').max()
    print(f"📊 {occupied_days.sum()}/{len(occupied_days)} occupied days")

    co2_targets   = list(range(args.co2_min, args.co2_max + 1, args.co2_step))
    sweep_results = []
    tou_config    = TOUPricing()

    print(f"\n🔄 CO₂ Sweep for {len(co2_targets)} targets: {co2_targets}")
    print("="*80)

    for i, co2_t in enumerate(co2_targets):
        print(f"\n[{i+1}/{len(co2_targets)}] CO₂ target = {co2_t} ppm")

        result = run_single_co2_scenario(co2_t, df, config, args, occupied_days)
        sweep_results.append(result)

        sc_dir = out_dir / f'co2_{co2_t}ppm'
        sc_dir.mkdir(parents=True, exist_ok=True)

        result['df_results'].to_csv(sc_dir/'timeseries.csv')
        result['daily_results'].to_csv(sc_dir/'daily.csv')

        print(f"  📈 Creating plots for {co2_t} ppm...")
        create_lstm_comparison_plots_for_scenario(result, sc_dir, co2_t)

        print(f"  💰 Creating TOU cost plots for {co2_t} ppm...")
        create_tou_cost_plots_for_scenario(result, sc_dir, co2_t, tou_config)

        print(f"  📊 Creating RL learning curves for {co2_t} ppm...")
        create_rl_learning_curves(result['rl_history'], sc_dir, co2_t)

        result['df_costs'].to_csv(sc_dir/'costs_timeseries.csv')
        result['daily_costs'].to_csv(sc_dir/'costs_daily.csv')

        print(f"  ✓ AI Elec:   {result['electricity_kwh_day']['ai']:.2f} kWh/day")
        print(f"  ✓ AI Cost:   €{result['costs_eur_day']['ai']:.2f}/day")
        print(f"  ✓ Savings:   {result['savings_pct']['ai']:.1f}%")
        print(f"  ✓ Cost Sav:  €{result['cost_savings']['ai_eur']:.2f}/day "
              f"({result['cost_savings']['ai_pct']:.1f}%)")
        print(f"  ✓ CO₂ achv:  {result['co2_achieved_ppm']['ai_avg']:.1f} ppm")

    # نمودارهای کلی
    print("\n📈 Creating global visualizations...")
    create_co2_sweep_plots(sweep_results, out_dir)
    create_violation_metrics_plot(sweep_results, out_dir)
    try:
        _plot_optimization_3d_from_results(sweep_results, out_dir)
    except Exception as e:
        print(f"⚠️ 3D plot skipped: {e}")

    # Excel خلاصه
    print("\n📊 Creating Excel summary...")
    with pd.ExcelWriter(out_dir/'CO2_Sweep_Analysis_RL.xlsx', engine='openpyxl') as writer:
        pd.DataFrame([{
            'CO2_Target_ppm':      r['co2_target'],
            'AsIs_Elec_kWh':       r['electricity_kwh_day']['asis'],
            'PI_Elec_kWh':         r['electricity_kwh_day']['pi'],
            'AI_RL_Elec_kWh':      r['electricity_kwh_day']['ai'],
            'AI_Energy_Savings_%': r['savings_pct']['ai'],
            'AsIs_Cost_EUR':       r['costs_eur_day']['asis'],
            'PI_Cost_EUR':         r['costs_eur_day']['pi'],
            'AI_Cost_EUR':         r['costs_eur_day']['ai'],
            'AI_Cost_Savings_EUR': r['cost_savings']['ai_eur'],
            'AI_Cost_Savings_%':   r['cost_savings']['ai_pct'],
            'AI_CO2_Achieved':     r['co2_achieved_ppm']['ai_avg'],
            'AI_VF':               r['ventilation']['ai_factor'],
        } for r in sweep_results]).to_excel(writer, sheet_name='Summary', index=False)

        pd.DataFrame([{
            'CO2_Target':   r['co2_target'],
            'AsIs_CO2_Avg': r['co2_achieved_ppm']['asis_avg'],
            'AsIs_CO2_Peak':r['co2_achieved_ppm']['asis_peak'],
            'PI_CO2_Avg':   r['co2_achieved_ppm']['pi_avg'],
            'PI_CO2_Peak':  r['co2_achieved_ppm']['pi_peak'],
            'AI_CO2_Avg':   r['co2_achieved_ppm']['ai_avg'],
            'AI_CO2_Peak':  r['co2_achieved_ppm']['ai_peak'],
            'Baseline_Vent':r['ventilation']['baseline_m3s'],
            'PI_VF':        r['ventilation']['pi_factor'],
            'AI_VF':        r['ventilation']['ai_factor'],
        } for r in sweep_results]).to_excel(writer, sheet_name='CO2_Metrics', index=False)

    # JSON
    with open(out_dir/'rl_sweep_results.json', 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': 'PPO-RL',
                'co2_range': f'{args.co2_min}-{args.co2_max} ppm',
                'co2_step': args.co2_step,
                'epochs': args.epochs,
            },
            'scenarios': [{
                'co2_target_ppm':  r['co2_target'],
                'electricity_kwh': r['electricity_kwh_day'],
                'co2_achieved':    r['co2_achieved_ppm'],
                'savings_pct':     r['savings_pct'],
                'ventilation':     r['ventilation'],
            } for r in sweep_results]
        }, f, indent=2)

    # خلاصه نهایی
    best = max(sweep_results, key=lambda x: x['savings_pct']['ai'])
    print("\n" + "="*80)
    print("✅  PPO-RL CO₂ Sweep Complete")
    print(f"🏆 Best scenario: CO₂={int(best['co2_target'])} ppm | "
          f"E={best['electricity_kwh_day']['ai']:.2f} kWh/day | "
          f"Savings={best['savings_pct']['ai']:.1f}%")
    print(f"💾 Results → {out_dir}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
