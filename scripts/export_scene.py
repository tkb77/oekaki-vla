"""Write the compiled drawing scene (from configs/env.yaml) to assets/scene_drawing.xml for inspection
(e.g. `python -m mujoco.viewer --mjcf assets/scene_drawing.xml`). The simulator itself builds the
scene from the config at runtime, so this file is informational only."""
from oekaki.config import load_config, resolve
from oekaki.env.scene import build_spec

if __name__ == "__main__":
    spec = build_spec(load_config())
    spec.compile()
    out = resolve("assets/scene_drawing.xml")
    # mesh paths are relative to assets/so101/ in the source model; the XML lives in assets/
    out.write_text(spec.to_xml().replace('meshdir="assets"', 'meshdir="so101/assets"', 1))
    print("wrote", out)
