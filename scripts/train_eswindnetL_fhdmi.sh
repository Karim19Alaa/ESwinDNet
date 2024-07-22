python eswindnet/train.py \
name=001_train_WSESwinDNetL_Demoiring_FHDMi_from_scratch \
model=eswindnet \
model.net.depth=6 \
model.net.num_heads=32 \
data=fhdmi \
data.opt.train.dataset.dataroot="datasets/fhdmi-dataset" \
data.opt.train.dataloader.batch_size=2 \
data.opt.val.dataset.dataroot="datasets/fhdmi-dataset/test" \
data.opt.test.dataset.dataroot="datasets/fhdmi-dataset/test" \
training.train.scheduler.eta_min=1e-5 \
+trainer.trainer_args.log_every_n_steps=100 \
+trainer.trainer_args.check_val_every_n_epoch=10 \
+trainer.trainer_args.num_sanity_val_steps=0 \
trainer.callbacks.EMAModelCheckpoint.save_top_k=1 \
trainer.trainer_args.max_steps=748651 