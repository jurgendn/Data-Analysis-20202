python cli.py \
    --model.model_name='dummy' \
    --model.optimizer_name='adam' \
    --model.lr=0.001 \
    --model.lr_scheduler_method_name='none' \
    --model.pretrained=true \
    --data.data_dir='data/' \
    --data.num_workers=4 \
    --data.batch_size=256 \
    --trainer.limit_train_batches=0.2 \
    --trainer.max_epochs=1 \
    --trainer.gpus=0 \
    --trainer.num_sanity_val_steps=2 \
    --trainer.default_root_dir='exps/dummy_adam' \
    --pickle_embedd