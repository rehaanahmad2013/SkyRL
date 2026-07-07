#!/usr/bin/env bash
# 2-node k8s launcher for fully-async CodeContests training.
# Runs as an Indexed Job (2 pods, 8 GPUs each):
#   rank 0: Ray head + trainer driver -> FSDP policy on its 8 GPUs
#   rank 1: Ray worker -> hosts the 8x TP1 vLLM engines
# Same command on every pod; behavior branches only on JOB_COMPLETION_INDEX.
set -exuo pipefail

cd "$(dirname "$0")/../../.."
REPO_ROOT="$PWD"

RANK="${JOB_COMPLETION_INDEX:-0}"
HEAD_HOST="${RAY_HEAD_HOST:-127.0.0.1}"
RAY_PORT=6379

# Pod network only (no IB exposed to these pods)
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
# First worker spawn on a fresh node builds the uv env (multi-GB download)
export RAY_worker_register_timeout_seconds=1200

command -v uv >/dev/null 2>&1 || pip install uv

UV_RUN=(uv run --isolated --extra fsdp --extra harbor)

# Task dirs must exist on every node: generation workers can land anywhere.
DATA_DIR="$HOME/data/harbor"
[ -d "$DATA_DIR/CodeContests" ] || \
  "${UV_RUN[@]}" examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests
[ -d "$DATA_DIR/OpenThoughts-TB-dev" ] || \
  "${UV_RUN[@]}" examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/OpenThoughts-TB-dev

TRAIN_DATA="['$DATA_DIR/CodeContests']"
EVAL_DATA="['$DATA_DIR/OpenThoughts-TB-dev']"

#-----------------------
# Worker rank: join the Ray cluster and stay up until the head goes away.
#-----------------------
if [ "$RANK" != "0" ]; then
  echo "[rank $RANK] waiting for ray head at $HEAD_HOST:$RAY_PORT"
  until (exec 3<>"/dev/tcp/$HEAD_HOST/$RAY_PORT") 2>/dev/null; do sleep 5; done
  "${UV_RUN[@]}" ray start --address="$HEAD_HOST:$RAY_PORT" --disable-usage-stats
  sleep 60
  while (exec 3<>"/dev/tcp/$HEAD_HOST/$RAY_PORT") 2>/dev/null; do sleep 15; done
  echo "[rank $RANK] ray head gone, exiting"
  exit 0
fi

#-----------------------
# Head rank: start Ray head, then run the trainer driver.
#-----------------------
"${UV_RUN[@]}" ray start --head --port=$RAY_PORT --dashboard-host=127.0.0.1 --disable-usage-stats

# Wait for the second node so placement groups don't race a 1-node cluster.
"${UV_RUN[@]}" python - <<'EOF'
import time, ray
ray.init(address="auto")
for _ in range(240):
    nodes = [n for n in ray.nodes() if n["Alive"]]
    gpus = sum(int(n["Resources"].get("GPU", 0)) for n in nodes)
    print(f"cluster: {len(nodes)} nodes, {gpus} GPUs", flush=True)
    if gpus >= 16:
        break
    time.sleep(5)
else:
    raise SystemExit("second node never joined the ray cluster")
EOF

RUN_NAME="codecontest-fullyasync-2node"
STORAGE_ROOT="/mnt/local_storage/$RUN_NAME"
TRIALS_DIR="$STORAGE_ROOT/trials_run"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

#-----------------------
# Training setup
#-----------------------
N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=16
MAX_MODEL_LEN=40960  # Qwen3-8B native max (up from 32768); YaRN beyond this is a separate variant

LOSS_REDUCTION="token_mean"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false
APPLY_OVERLONG_FILTERING=true

CHAT_TEMPLATE_PATH="$REPO_ROOT/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

SEQUENCE_MASK_METRIC=geometric
GEO_MASK_HIGH=1.01
GEO_MASK_LOW=0.99

# Constraint: mini_batch_size <= num_parallel_generation_workers <= mini_batch_size * (max_staleness_steps + 1)
MAX_STALENESS_STEPS=4
NUM_PARALLEL_GENERATION_WORKERS=$(( MINI_BATCH_SIZE * 4 ))

#----------------
# 2x8 RTX PRO 6000 Blackwell (96GB, no NVLink):
# trainer node = 8 policy GPUs; inference node = 8 single-GPU engines.
#----------------
NUM_INFERENCE_ENGINES=8
TP_SIZE=1
NUM_POLICY_GPUS=8
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=5
MAX_CONCURRENCY=128

"${UV_RUN[@]}" -m examples.train_integrations.harbor.entrypoints.main_harbor_fully_async \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.flash_attn=false \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=$MAX_STALENESS_STEPS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  trainer.algorithm.policy_loss_type="rollout_is" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.off_policy_correction.sequence_mask_metric=$SEQUENCE_MASK_METRIC \
  trainer.algorithm.off_policy_correction.geo_mask_high=$GEO_MASK_HIGH \
  trainer.algorithm.off_policy_correction.geo_mask_low=$GEO_MASK_LOW \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=2 \
  trainer.hf_save_interval=25 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=2 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.9 \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.inference_engine.enforce_eager=false \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY
