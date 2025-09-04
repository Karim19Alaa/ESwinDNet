python eswindnet/test.py \
name=001_train_WSESwinDNet_Demoiring_FHDMi_test \
data=uhdm \
model=eswindnet \
testing=testing_fhdmi \
data.opt.train.dataset.dataroot="datasets/fhdmi-dataset" \
data.opt.val.dataset.dataroot="datasets/fhdmi-dataset/test" \
data.opt.test.dataset.dataroot="datasets/fhdmi-dataset/test" \
testing.trainer_args.ckpt_path= # {CHANGEME}