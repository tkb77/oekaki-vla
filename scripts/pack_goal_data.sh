#!/usr/bin/env bash
# Pack dataset goal_v1 for the training machine (run on the simulation machine): data/goal_v1 -> goal_v1.tar
# On the training machine: tar xf goal_v1.tar   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."
tar cf goal_v1.tar data/goal_v1
ls -la goal_v1.tar
