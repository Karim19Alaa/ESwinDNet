python eswindnet/test.py \
name=001_train_WSESwinDNet_Demoiring_UHDM_test \
data=uhdm \
model=eswindnet \
testing=testing_uhdm \
data.opt.train.dataset.dataroot="datasets/uhdm/train" \
data.opt.val.dataset.dataroot="datasets/uhdm/test" \
data.opt.test.dataset.dataroot="datasets/uhdm/test" \
testing.trainer_args.ckpt_path= # {CHANGEME}