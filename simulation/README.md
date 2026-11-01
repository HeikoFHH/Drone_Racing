# Isaac Lab — two-pole Tello race

Simulated outdoor race for **drone-race-hacks**: one Tello, Pole A → Pole B → Pole A, dirt-field-inspired flat ground. Train with **PPO** (Stable-Baselines3) before flying the real course by Harvard i-labs.

| | |
|---|---|
| Gym id | `AS-Tello-Race-v0` |
| Package | `simulation/source/drone_race_kit/` |
| Lap rules | Same as `docs/RACE_RULES.md` / `race/lap_timer.py` |
| Real venue | Outdoor dirt field / parking lot (`presentation/assets/dirt_field.png`) |

## Prerequisites

1. **Isaac Sim 5.x + Isaac Lab** (same stack as Action Space). On Jaime’s box that is typically `~/isaac/` with venv `~/isaac/env_isaaclab`.
2. **Action Space kit** (Tello asset + velocity controller):

   ```bash
   # from your Action_Space checkout
   pip install -e simulation_infra/source/action_space_kit
   ```

3. **This race kit** (from `drone-race-hacks`):

   ```bash
   cd /path/to/drone-race-hacks
   # use the Isaac Lab venv, not the laptop-only .venv
   source ~/isaac/env_isaaclab/bin/activate
   unset CONDA_PREFIX CONDA_DEFAULT_ENV
   export OMNI_KIT_ACCEPT_EULA=YES
   export TELLO_USD_PATH=$HOME/isaac/assets/tello.usd   # required
   pip install -e simulation/source/drone_race_kit
   ```

Import once to register the task:

```bash
python -c "import drone_race_kit; import gymnasium as gym; print(gym.spec('AS-Tello-Race-v0'))"
```

## Train (PPO)

Reuse Action Space’s SB3 trainer (or Isaac Lab’s):

```bash
cd /path/to/Action_Space/simulation_infra   # or copy scripts/sb3/train.py locally
python -u scripts/sb3/train.py --task AS-Tello-Race-v0 --headless \
  --num_envs 512 --max_iterations 1000
```

Agent hyperparams: `drone_race_kit/tasks/direct/tello_race/agents/sb3_ppo_cfg.yaml`.

Helper wrapper (if Action_Space lives next to this repo):

```bash
./simulation/scripts/train_race.sh --num_envs 256 --max_iterations 200
```

## Play / smoke

```bash
python -u scripts/sb3/play.py --task AS-Tello-Race-v0 --num_envs 16 \
  --checkpoint /path/to/model.zip
```

Or:

```bash
./simulation/scripts/play_race.sh --checkpoint /path/to/model.zip
```

## Design notes

- **Actions:** 4× SDK-style rc channels in `[-1, 1]` via `TelloVelocityController` (same interface as hover/room).
- **Observations:** vectors to Pole A/B, velocities, gravity, lap-phase one-hot, elapsed lap time.
- **Rewards:** progress to next gate, phase bonuses (leave A / hit B / finish), light time penalty while racing, crash / tilt costs.
- **Ground:** tan flat plane — placeholder for the dirt field; swap USD later if you want photoreal Allston.

## Without a GPU box

You can still develop the real-world stack (`core/`, `race/`, presentation) on a laptop. Sim training needs Isaac Lab + NVIDIA GPU.

> Mac / no-GPU alternative: see [`drone_race_sim/`](../drone_race_sim/README.md) and [`docs/SIMULATION_PLAN.md`](../docs/SIMULATION_PLAN.md).
