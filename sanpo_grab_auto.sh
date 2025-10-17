#!/usr/bin/env bash
set -euo pipefail

# knobs (override at run time like: BUDGET_GB=10 KEEP_FRAMES=300 bash sanpo_grab_auto_fixed.sh)
BUDGET_GB=${BUDGET_GB:-20}         # max local size
KEEP_FRAMES=${KEEP_FRAMES:-400}    # max files per type per session
PREFIX="gs://gresearch/sanpo_dataset/v0/sanpo-real"
DEST="./sanpo_subset_auto"

mkdir -p "$DEST"

echo "[*] Enumerating sessions…"
mapfile -t RAW < <(gsutil ls "${PREFIX}/")
# strip the _$folder$ placeholders and ensure trailing slash
mapfile -t SESS < <(printf "%s\n" "${RAW[@]}" | sed -n "s#${PREFIX}/##p" | sed 's/_$folder$//' | sed 's#/*$#/#' | sort -u)

echo "[*] Found ${#SESS[@]} sessions."
echo "[*] Download root: $DEST"

pick_and_copy () {
  local sess="$1" kind="$2" hint_regex="$3" exts_regex="$4"
  local root="${PREFIX}/${sess}"
  local out="${DEST}/sanpo-real/sessions/${sess}${kind}"
  mkdir -p "$out"

  # cache a recursive listing once per session (no wildcards → no warnings)
  local cache="/tmp/ls_${sess//\//}.txt"
  if [[ ! -f "$cache" ]]; then
    echo "[-] Listing ${root} (this may take a bit)…"
    gsutil ls -r "$root" > "$cache"
  fi

  # prefer hinted paths; if none match, fall back to any files with the extensions
  local list="/tmp/${kind}_${sess//\//}.txt"
  grep -Ei "/${hint_regex}/.*\.${exts_regex}$" "$cache" \
    | head -n "${KEEP_FRAMES}" > "$list" || true
  if [[ ! -s "$list" ]]; then
    grep -Ei "\.(${exts_regex})$" "$cache" \
      | head -n "${KEEP_FRAMES}" > "$list" || true
  fi

  local n=$(wc -l < "$list" || echo 0)
  if [[ "$n" -gt 0 ]]; then
    echo "    -> copying $n ${kind}"
    gsutil -m cp -I "$out/" < "$list" || true
  else
    echo "    -> no ${kind} matched"
  fi
  rm -f "$list"
}

for s in "${SESS[@]}"; do
  echo "[*] Session: ${s}"
  # images: prefer left-like camera dirs; fall back to any images
  pick_and_copy "$s" "images" "(left|cam0|camera_left|stereo_left|rgb_left|cam/0|camera/0)" "(jpg|jpeg|png|bmp|webp)"
  # masks: look for common label dirs; include npy/npz as possible mask formats
  pick_and_copy "$s" "masks" "(mask|seg|panoptic|semantic|label|annot|anno)" "(png|webp|bmp|jpg|jpeg|npy|npz)"

  # light metadata if present
  gsutil -m cp "${PREFIX}/${s}"*.{csv,json} "${DEST}/sanpo-real/sessions/${s%/}" 2>/dev/null || true

  # budget check (fast, local)
  USED_BYTES=$(du -sb "$DEST" | awk '{print $1}')
  USED_GB=$(awk -v b="$USED_BYTES" 'BEGIN{printf "%.2f", b/1024/1024/1024}')
  echo "    [size] ~${USED_GB} GB so far"
  awk -v u="$USED_GB" -v m="$BUDGET_GB" 'BEGIN{exit !(u>m)}' && { echo "[!] Budget hit, stopping."; break; }
done

echo "[✓] Done. Final size:"; du -sh "$DEST" || true
