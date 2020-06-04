#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate

python train.py --do_test \
    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
    --test_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_test.balanced.in \
    --test_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_test.balanced.label.bin.out \
    --load_model=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/cls_output/bi_att  \
    --classification --classification_model=RNN --output_classes=2 \
    --enc_type=bi --enc_num_units=256 --cls_attention --cls_attention_size=50 \
    --learning_rate=0.001 --batch_size=32 --max_len=50 \
    --num_epochs=10 --print_every_steps=100 --stop_steps=5000 \
    --output_dir=cls_output_test \
    --save_checkpoints \
    --num_gpus=0


## Test against augmented classifier from the AE+LS+CF
#python train.py --do_test \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --test_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_test.balanced.in \
#    --test_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_test.balanced.label.bin.out \
#    --load_model=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/adv_output_lscf/nmt-T2.ckpt  \
#    --classification --classification_model=RNN --output_classes=2 \
#    --enc_type=bi --enc_num_units=256 --cls_attention --cls_attention_size=50 \
#    --learning_rate=0.001 --batch_size=32 --max_len=50 \
#    --num_epochs=10 --print_every_steps=100 --stop_steps=5000 \
#    --output_dir=cls_output_test \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --use_defending_as_target

