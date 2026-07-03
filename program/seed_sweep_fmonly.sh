#!/usr/bin/env bash
# PAPER: Result 8 -- seed sweep (42/123/7) for tab:fmonly. See RESULTS_TO_CODE.md
# Seed sweep for tab:fmonly: adds seeds 123 and 7 to the existing seed 42.
# Three conditions per seed: logit-only, fm+logit, fm-only.
# All CE-free (--ce_w 0.0 --kd_w 1.0), pure soft-KD, matching the seed-42 run.
#
# Run from ~/SiHan/new_loss with the SAME train_kd.py / corpus.jsonl used for seed 42.
# Verify first:  grep -n "skipped" train_kd.py   (must be the malformed-line-tolerant version)

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER=Qwen/Qwen2.5-1.5B
STUDENT=Qwen/Qwen2.5-0.5B
DATA=corpus.jsonl
COMMON="--teacher $TEACHER --student $STUDENT --data $DATA --lr 2e-5 --ce_w 0.0 --kd_w 1.0 --steps 2000 --bs 1 --seqlen 512 --accum 8"

for SEED in 123 7; do
  echo "############## SEED $SEED : logit only ##############"
  python -u train_kd.py --mode logit $COMMON --seed $SEED --save student_pureKD_s${SEED}.pt

  echo "############## SEED $SEED : fm + logit ##############"
  python -u train_kd.py --mode fm    $COMMON --seed $SEED --save student_pureKD_fm_s${SEED}.pt

  echo "############## SEED $SEED : fm only ##############"
  python -u train_kd.py --mode fm --fm_only $COMMON --seed $SEED --save student_fmonly_s${SEED}.pt
done

echo "############## EVAL: all seeds, all conditions ##############"
PARQUET=/home/quantadft/models/hub/datasets--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/validation-00000-of-00001.parquet

HF_HOME=/home/quantadft/models HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -u eval_ppl.py --student $STUDENT --seqlen 512 --base-only \
  --parquet "$PARQUET" \
  --ckpts \
    student_pureKD_lr2e5.pt      student_pureKD_fm_lr2e5.pt      student_fmonly_lr2e5.pt \
    student_pureKD_s123.pt       student_pureKD_fm_s123.pt       student_fmonly_s123.pt \
    student_pureKD_s7.pt         student_pureKD_fm_s7.pt         student_fmonly_s7.pt \
  --max-rows 2000
