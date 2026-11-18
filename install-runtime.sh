#!/usr/bin/env bash
# Installs the Python runtime and models used by Ally Sound Lab.
# Run as the regular desktop user (not root). Needs curl and sha256sum;
# uv is used from the system if present, otherwise a pinned copy is fetched.
# Lines starting with "## " are progress steps shown in the plugin menu.
set -euo pipefail

DATA="${HOME}/.local/share/ally-sound-npu"
UV_VERSION=0.13.6
UV_SHA256=f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645
mkdir -p "${DATA}/models" "${DATA}/bin"
cd "${DATA}"

check() {  # check SHA256 FILE
    echo "$1  $2" | sha256sum -c --status -
}

echo "## 1/4 Инструмент установки (uv)"
# ASL_UV="" forces the pinned download (used to test a clean machine).
UV=${ASL_UV-$(command -v uv || true)}
if [ -z "${UV}" ]; then
    UV="${DATA}/bin/uv"
    if [ ! -x "${UV}" ]; then
        archive="${DATA}/bin/uv.tar.gz"
        curl -fL --retry 2 -o "${archive}" \
            "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
        check "${UV_SHA256}" "${archive}" || { echo "## Ошибка: контрольная сумма uv не совпала"; exit 1; }
        tar -xzf "${archive}" -C "${DATA}/bin" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
        rm -f "${archive}"
    fi
fi

echo "## 2/4 Python 3.13"
if [ ! -x .venv/bin/python ]; then
    "${UV}" venv --python 3.13 .venv
fi

echo "## 3/4 Библиотеки (numpy, scipy, onnxruntime, h5py)"
"${UV}" pip install --python .venv/bin/python \
    numpy==2.5.5 scipy==1.18.1 onnxruntime==1.32.0 h5py==3.16.0

fetch() {
    local name=$1 url=$2 sum=$3
    if [ -f "models/${name}" ] && check "${sum}" "models/${name}"; then
        return
    fi
    curl -fL --retry 2 -o "models/${name}.part" "${url}"
    check "${sum}" "models/${name}.part" || { echo "## Ошибка: контрольная сумма ${name} не совпала"; exit 1; }
    mv "models/${name}.part" "models/${name}"
}

echo "## 4/4 Модели (StemgenRT, HRTF KU100)"
# StemgenRT streaming vocal separator.
fetch vocals.onnx \
    https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/main/model/parallel/vocals.onnx \
    f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148
# SADIE II KU100 dummy-head HRIRs (48 kHz).
fetch ku100.sofa \
    https://zenodo.org/records/12542676/files/SADIE_KU100_DFC_256_order_fir_48000.sofa \
    6822b4318ce80ab6b8421a4b55b1edcedde1b276426aa5416526a57cccbe7e58

echo "## Готово"
