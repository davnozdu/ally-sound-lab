#!/usr/bin/env bash
# Installs the Python runtime and models used by Ally Sound Lab.
# Run as the regular desktop user (not root). Requires: uv, curl, sha256sum.
set -euo pipefail

DATA="${HOME}/.local/share/ally-sound-npu"
mkdir -p "${DATA}/models"
cd "${DATA}"

if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.13 .venv
fi
uv pip install --python .venv/bin/python \
    numpy==2.5.5 scipy==1.18.1 onnxruntime==1.32.0 h5py==3.16.0

fetch() {
    local name=$1 url=$2 sum=$3
    if [ -f "models/${name}" ] && echo "${sum}  models/${name}" | sha256sum -c --status -; then
        echo "${name}: already installed"
        return
    fi
    curl -fL --retry 2 -o "models/${name}.part" "${url}"
    echo "${sum}  models/${name}.part" | sha256sum -c -
    mv "models/${name}.part" "models/${name}"
}

# StemgenRT streaming vocal separator.
fetch vocals.onnx \
    https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/main/model/parallel/vocals.onnx \
    f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148
# SADIE II KU100 dummy-head HRIRs (48 kHz).
fetch ku100.sofa \
    https://zenodo.org/records/12542676/files/SADIE_KU100_DFC_256_order_fir_48000.sofa \
    6822b4318ce80ab6b8421a4b55b1edcedde1b276426aa5416526a57cccbe7e58

echo "Runtime ready in ${DATA}"
