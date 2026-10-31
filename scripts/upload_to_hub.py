"""Upload the trained models and the dataset to the Hugging Face Hub, with the cards in docs/hub/.

  <namespace>/smolvla-oekaki-goal      SmolVLA model G (outputs/train/goal_G/checkpoints/last)
  <namespace>/oekaki-sketch-diffusion  the line-drawing generator (outputs/generator/last.pt)
  <namespace>/oekaki-goal-v1           dataset goal_v1 (data/goal_v1/G, LeRobot v3.0, tagged for LeRobot)

Usage (after `hf auth login`; the namespace defaults to the logged-in account):
  python scripts/upload_to_hub.py --code-url https://github.com/<user>/oekaki-vla [--private] \
      [--only policy generator dataset] [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib

from oekaki.config import resolve

REPOS = {
    "policy": dict(name="smolvla-oekaki-goal", type="model", card="docs/hub/policy.md",
                   folder="outputs/train/goal_G/checkpoints/last",
                   extra=["docs/img/robot_drawing.png", "docs/img/goal_demo_strip.png"]),
    "generator": dict(name="oekaki-sketch-diffusion", type="model", card="docs/hub/generator.md",
                      folder="outputs/generator", allow=["last.pt", "config.json"],
                      extra=["results/goal/generator/samples_post.png",
                             "results/goal/generator/trajectory_star.png"]),
    "dataset": dict(name="oekaki-goal-v1", type="dataset", card="docs/hub/dataset.md",
                    folder="data/goal_v1/G", ignore=["images/**"], extra=[], tag="v3.0"),
}


def card(spec: dict, namespace: str, code_url: str) -> str:
    text = resolve(spec["card"]).read_text()
    return text.replace("{{NAMESPACE}}", namespace).replace("{{CODE_URL}}", code_url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-url", required=True, help="URL of the code repository, linked from the cards")
    ap.add_argument("--namespace", default=None, help="Hub user or organisation (default: the logged-in account)")
    ap.add_argument("--only", nargs="+", choices=list(REPOS), default=list(REPOS))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    api = HfApi()
    ns = args.namespace or api.whoami()["name"]
    for key in args.only:
        spec = REPOS[key]
        repo, rtype, folder = f"{ns}/{spec['name']}", spec["type"], resolve(spec["folder"])
        files = sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())
        if spec.get("allow"):
            files = [f for f in files if str(f) in spec["allow"]]
        if spec.get("ignore"):
            files = [f for f in files if not any(f.match(g) for g in spec["ignore"])]
        size = sum((folder / f).stat().st_size for f in files) / 1e6
        print(f"{rtype} {repo}: {len(files)} files, {size:.0f} MB from {spec['folder']} + README.md + {spec['extra']}")
        if args.dry_run:
            continue
        api.create_repo(repo, repo_type=rtype, private=args.private, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=folder,
                          allow_patterns=spec.get("allow"), ignore_patterns=spec.get("ignore"),
                          commit_message=f"Upload {spec['name']}")
        for path in spec["extra"]:
            api.upload_file(path_or_fileobj=resolve(path), path_in_repo=resolve(path).name, repo_id=repo,
                            repo_type=rtype, commit_message=f"Add {resolve(path).name}")
        api.upload_file(path_or_fileobj=card(spec, ns, args.code_url).encode(), path_in_repo="README.md",
                        repo_id=repo, repo_type=rtype, commit_message="Add card")
        if spec.get("tag"):  # LeRobot looks the dataset up by its codebase-version tag
            with contextlib.suppress(RevisionNotFoundError):
                api.delete_tag(repo, tag=spec["tag"], repo_type=rtype)
            api.create_tag(repo, tag=spec["tag"], repo_type=rtype)
        print("  ->", f"https://huggingface.co/{'datasets/' if rtype == 'dataset' else ''}{repo}")


if __name__ == "__main__":
    main()
