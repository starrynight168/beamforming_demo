from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STANDARD_SCENE_IDS = (
    "simulation_contrast_speckle",
    "simulation_resolution_distortion",
    "experiments_contrast_speckle",
    "experiments_resolution_distortion",
)
RESOLUTION_SCENE_SOURCES = {
    "simulation_resolution_distortion": "simulation",
    "experiments_resolution_distortion": "experiments",
}
SCENE_LABELS = {
    "simulation_contrast_speckle": "Sim. contrast",
    "simulation_resolution_distortion": "Sim. resolution",
    "experiments_contrast_speckle": "Exp. contrast",
    "experiments_resolution_distortion": "Exp. resolution",
}


def resolve_project_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_scenes(config_path, scene_ids, *, sources=None):
    with resolve_project_path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    scenes = config.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("config.yaml must contain a scenes list")
    by_id = {str(scene.get("id")): scene for scene in scenes if isinstance(scene, dict)}
    missing = [scene_id for scene_id in scene_ids if scene_id not in by_id]
    if missing:
        kind = "resolution" if sources is not None else "standard"
        raise ValueError(f"config.yaml is missing {kind} scenes: {missing}")
    selected = []
    for scene_id in scene_ids:
        scene = by_id[scene_id]
        data_path = resolve_project_path(scene["h5_path"])
        if not data_path.is_file():
            raise FileNotFoundError(f"{scene_id}: H5 file not found: {data_path}")
        item = {"id": scene_id, "h5_path": data_path,
                "sample_idx": int(scene.get("sample_idx", 0))}
        if sources is not None:
            item["expected_source"] = sources[scene_id]
        selected.append(item)
    return selected
