CUDA_VISIBLE_DEVICES=7 python scripts/rsl_rl/train_ours.py --task=Tracking-Flat-G1-v0 \
--motion_root /media/raid/workspace/huangyuming/TWIST2/TWIST2_full_npz/ \
--headless --logger wandb --log_project_name deepmimic --run_name 20260331-Tracking-Flat-G1-v0 \
obs_pipeline.mode=twist2_like \
obs_pipeline.include_history=true \
obs_pipeline.history_len=10 \
obs_pipeline.include_future=false
