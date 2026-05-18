#!/usr/bin/env bash
# Add Diffusion Policy and LOVE as git submodules under third_party/.
# Run once from repo root:  bash scripts/setup_submodules.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DP_URL="https://github.com/real-stanford/diffusion_policy.git"
LOVE_URL="https://github.com/yidingjiang/love.git"
DP_PATH="third_party/diffusion_policy"
LOVE_PATH="third_party/love"

remove_placeholder() {
  local path="$1"
  if [[ -d "$path" ]]; then
    rm -f "$path/.gitkeep"
    # submodule add requires missing or empty path
    if [[ -z "$(ls -A "$path" 2>/dev/null)" ]]; then
      rmdir "$path" 2>/dev/null || true
    elif [[ -f "$path/.gitkeep" ]] || [[ "$(ls -A "$path")" == ".gitkeep" ]]; then
      rm -rf "$path"
    fi
  fi
}

if [[ -f .gitmodules ]]; then
  echo "Updating existing submodules..."
  git submodule update --init --recursive
  exit 0
fi

remove_placeholder "$DP_PATH"
remove_placeholder "$LOVE_PATH"

echo "Adding diffusion_policy..."
git submodule add "$DP_URL" "$DP_PATH"

echo "Adding love..."
git submodule add "$LOVE_URL" "$LOVE_PATH"

git submodule update --init --recursive

echo ""
echo "Done. Submodule status:"
git submodule status
echo ""
echo "Optional: pin a specific commit in each submodule, then commit from repo root:"
echo "  cd $DP_PATH && git checkout <sha> && cd ../.."
echo "  cd $LOVE_PATH && git checkout <sha> && cd ../.."
echo "  git add .gitmodules $DP_PATH $LOVE_PATH"
