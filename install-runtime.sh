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

install_yamnet() (
    local dest="models/yamnet"
    local graph_sum=970ea63f4dc1aa33c86ef7d61d854319fdfe6bdd078f5b3d674ee1db5dd96471
    local data_sum=1d6dff488da4712190e10f1c841252aa58011381132d0126dcd5ee56a4e7ed31
    if [ -f "$dest/yamnet.onnx" ] && [ -f "$dest/yamnet.data" ] &&
        check "$graph_sum" "$dest/yamnet.onnx" && check "$data_sum" "$dest/yamnet.data"; then
        echo "## YAMNet готова"
        return
    fi
    echo "## Скачивание модели YAMNet"
    local temporary
    temporary=$(mktemp -d models/.yamnet.XXXXXX)
    trap 'rm -rf -- "$temporary"' EXIT
    curl -fL --retry 2 -o "$temporary/model.zip" \
        https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/yamnet/releases/v0.59.0/yamnet-onnx-float.zip
    check bda628207418bc59b6a371aef6180b53367707eaf7b8b8a7c683417000d1eea4 "$temporary/model.zip" ||
        { echo "## Ошибка: контрольная сумма YAMNet не совпала"; exit 1; }
    .venv/bin/python - "$temporary" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile

target = Path(sys.argv[1])
with ZipFile(target / "model.zip") as archive:
    for name in ("yamnet.onnx", "yamnet.data"):
        (target / name).write_bytes(archive.read("yamnet-onnx-float/" + name))
PY
    check "$graph_sum" "$temporary/yamnet.onnx" && check "$data_sum" "$temporary/yamnet.data" ||
        { echo "## Ошибка: файлы модели YAMNet повреждены"; exit 1; }
    mkdir -p "$dest"
    mv -f "$temporary/yamnet.data" "$temporary/yamnet.onnx" "$dest/"
    echo "## YAMNet готова"
)

if [ "${1:-}" = "--auto-eq" ]; then
    install_yamnet
    # Discogs-EffNet (Essentia, 400 Discogs styles) replaced MSD-MusiCNN for the genre.
    rm -f models/yamnet/msd-musicnn-1.onnx
    genre="models/yamnet/discogs-effnet-bsdynamic-1.onnx"
    genre_sum=a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c
    if [ ! -f "$genre" ] || ! check "$genre_sum" "$genre"; then
        echo "## Скачивание модели Discogs-EffNet"
        temporary=$(mktemp models/.effnet.XXXXXX)
        trap 'rm -f -- "$temporary"' EXIT
        curl -fL --retry 2 -o "$temporary" \
            https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx
        check "$genre_sum" "$temporary" ||
            { echo "## Ошибка: контрольная сумма Discogs-EffNet не совпала"; exit 1; }
        mv -f "$temporary" "$genre"
    fi
    echo "## Модели автоэквалайзера готовы"
    exit 0
fi

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
