#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate

python train.py --do_train \
    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
    --enc_type=bi --attention --enc_num_units=512 --dec_num_units=512 \
    --learning_rate=0.001 --batch_size=32 --max_len=50 \
    --num_epochs=10 --print_every_steps=100 --stop_steps=20000 \
    --output_dir=ae_output \
    --save_checkpoints \
    --num_gpus=0
