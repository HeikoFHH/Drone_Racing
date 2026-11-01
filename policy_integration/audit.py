"""Read-only integration audit of the upstream engine, timer, and simulator."""

import argparse
import ast
import json
from pathlib import Path


def assignments(path):
    tree = ast.parse(path.read_text())
    return {node.targets[0].id: node.value for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}


def audit(root):
    import yaml  # Already an upstream requirement; no camera/flight imports.
    tags = yaml.safe_load((root / "config/tags.yaml").read_text())
    configured = set(int(x) for x in tags["sizes_mm"])
    legacy = assignments(root / "core/tello_tags.py")
    legacy_ids = set(ast.literal_eval(key) for key in legacy["TAG_SIZE_M"].keys)
    engine = assignments(root / "core/tello_dual_video.py")
    fallback_id = ast.literal_eval(engine["TAG_ID"])
    findings = []
    if not configured.issubset(legacy_ids | {fallback_id}):
        findings.append({"id": "tag_config_split", "severity": "block_live_handoff",
                         "evidence": {"configured_tags": sorted(configured), "engine_tags": sorted(legacy_ids), "fallback_tag": fallback_id},
                         "next": "Sensing/runner owner must use one measured tag map and size registry."})
    if not (root / "core/tag_map.json").exists():
        findings.append({"id": "legacy_map_absent", "severity": "block_live_handoff",
                         "evidence": "engine imports tello_tags.load_map; its default path is core/tag_map.json, not config/map.yaml",
                         "next": "Wire Positioner with config/map.yaml or explicitly migrate the legacy map path."})
    timer = (root / "race/lap_timer.py").read_text()
    sim = (root / "simulation/source/drone_race_kit/drone_race_kit/tasks/direct/tello_race/tello_race_env.py").read_text()
    if "p[2] - q[2]" in timer and "pos[:, :2]" in sim:
        findings.append({"id": "coordinate_plane_mismatch", "severity": "requires_explicit_adapter",
                         "evidence": "lap timer uses map X/Z; simulator gates use world X/Y",
                         "next": "Apply verified map->course rigid transform before the policy; transform back for existing lap timer."})
    train = (root / "simulation/scripts/train_race.sh").read_text()
    if 'python -c "import drone_race_kit"' in train:
        findings.append({"id": "training_process_registration", "severity": "partner_verification",
                         "evidence": "wrapper imports task in a separate Python process before executing external trainer",
                         "next": "Verify the actual training process imports drone_race_kit before gym.make; a previous process cannot register it."})
    findings.append({"id": "ppo_real_observation_gap", "severity": "block_rl_handoff",
                     "evidence": "PPO expects 21 features including angular velocity/body gravity; new canonical policy packet has position, linear velocity and yaw",
                     "next": "RL owner must provide measured/estimated features and exact preprocessing, or retrain with the common sensor-realistic input."})
    return {"audit_scope": "static source only; not hardware verification", "findings": findings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.repo), indent=2))
    # A completed audit is not flight readiness. Findings remain in the report.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
