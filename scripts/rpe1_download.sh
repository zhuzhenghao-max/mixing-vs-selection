#!/bin/bash
# RPE1 h5ad resumable 4-way range download (same structure as gwps_download.sh).
cd "$(dirname "$0")/.." || exit 1
URL="https://exampledata.scverse.org/pertpy/replogle_2022_rpe1.h5ad"
T=1236886900
C=$(( (T+3)/4 ))
mkdir -p data/external/rpe1

dl_part() {
  local i=$1 S=$2 E=$3 out=$4
  local want=$((E-S+1)) attempt have
  for attempt in $(seq 1 60); do
    have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$have" -ge "$want" ]; then echo "part$i COMPLETE $have/$want"; return 0; fi
    echo "part$i attempt $attempt: have $have/$want, range $((S+have))-$E"
    curl -sL --ssl-no-revoke --connect-timeout 20 --speed-limit 200000 --speed-time 30 \
         --retry 5 --retry-delay 3 -r $((S+have))-$E -o "$out.tmp" "$URL"
    if [ -s "$out.tmp" ]; then
      { cat "$out" 2>/dev/null; cat "$out.tmp"; } > "$out.new"
      mv -f "$out.new" "$out"
      rm -f "$out.tmp"
    fi
  done
  echo "part$i FAILED after 60 attempts"; return 1
}

for i in 0 1 2 3; do
  S=$((i*C)); E=$(( (i+1)*C - 1 )); if [ $E -ge $T ]; then E=$((T-1)); fi
  dl_part $i $S $E "data/external/rpe1/part$i" &
done
wait
echo "ALL_PARTS_DONE"
ok=1
for i in 0 1 2 3; do
  S=$((i*C)); E=$(( (i+1)*C - 1 )); if [ $E -ge $T ]; then E=$((T-1)); fi
  want=$((E-S+1)); have=$(stat -c%s "data/external/rpe1/part$i" 2>/dev/null || echo 0)
  echo "part$i $have/$want"
  [ "$have" -ge "$want" ] || ok=0
done
if [ "$ok" = 1 ]; then
  cat data/external/rpe1/part0 data/external/rpe1/part1 \
      data/external/rpe1/part2 data/external/rpe1/part3 \
      > data/external/rpe1/replogle_2022_rpe1.h5ad
  sz=$(stat -c%s data/external/rpe1/replogle_2022_rpe1.h5ad)
  echo "assembled size $sz (expect $T)"
  if [ "$sz" = "$T" ]; then
    rm -f data/external/rpe1/part0 data/external/rpe1/part1 \
          data/external/rpe1/part2 data/external/rpe1/part3
    echo "RPE1_READY"
  else
    echo "SIZE_MISMATCH"
  fi
else
  echo "PARTS_INCOMPLETE"
fi
