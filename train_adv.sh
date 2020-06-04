#!/usr/bin/env bash

source /Users/yxu132/pyflow3.6/bin/activate

# AE+bal
python train.py --do_train \
    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-cls/bi_att  \
    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att \
    --adv --classification_model=RNN  --output_classes=2 --balance \
    --gumbel_softmax_temporature=0.1  \
    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
    --cls_attention --cls_attention_size=50 --attention \
    --batch_size=16 --max_len=50 \
    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
    --learning_rate=0.0001 --ae_lambda=0.2 --seq_lambda=0.7  \
    --output_dir=adv_output_bal \
    --save_checkpoints \
    --num_gpus=0


## AE+LS
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
#    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
#    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
#    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
#    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-cls/bi_att  \
#    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att \
#    --adv --classification_model=RNN  --output_classes=2 \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=adv_output_ls \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95
#
#
#
## AE+LS+GAN
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
#    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
#    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
#    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
#    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-cls/bi_att  \
#    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att \
#    --adv --classification_model=RNN  --output_classes=2 \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=adv_output_lsgan \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95 \
#    --gan --at_steps=2
#
#
## AE+LS+CF
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
#    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
#    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
#    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
#    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-cls/bi_att  \
#    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=adv_output_lscf \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_vocab.in  \
#	--ae_emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_emb.json
#
## AE+LS+CF+CPY
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
#    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
#    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
#    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
#    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-cls/bi_att  \
#    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=adv_output_lscfcp \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_vocab.in  \
#	--ae_emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_emb.json \
#	--copy --attention_copy_mask --use_stop_words --top_k_attack=9
#
## Conditional PTN: AE+LS+CF
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/train.pos.in \
#    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/train.pos.out \
#    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/dev.pos.in \
#    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/dev.pos.out \
#    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-cls/bi_att  \
#    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=adv_output_lscfcp_ptn \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_vocab.in  \
#	 --ae_emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_emb.json \
#	 --copy --attention_copy_mask --use_stop_words --top_k_attack=9 \
#    --target_label=0
#
#
# AE+LS+CF+DEFENCE
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
#    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
#    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
#    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
#    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-cls/bi_att  \
#    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2 \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0 \
#    --output_dir=def_output \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_vocab.in  \
#	--ae_emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_emb.json \
#    --defending --at_steps=2
#
#
## Attacking an augmented AE+LS+CF model: AE+LS+CF
#python train.py --do_train \
#    --vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in \
#    --emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json \
#    --input_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in \
#    --output_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out \
#    --dev_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.in \
#    --dev_output=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out \
#    --load_model_ae=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/yelp50_x3-ae/bi_att_cf_fixed \
#    --adv --classification_model=RNN  --output_classes=2  \
#    --gumbel_softmax_temporature=0.1  \
#    --enc_type=bi --cls_enc_num_units=256 --cls_enc_type=bi \
#    --cls_attention --cls_attention_size=50 --attention \
#    --batch_size=16 --max_len=50 \
#    --num_epochs=20 --print_every_steps=100 --total_steps=200000 \
#    --learning_rate=0.00001 --ae_lambda=0.8 --seq_lambda=1.0  \
#    --output_dir=adv_aeaug_lscf \
#    --save_checkpoints \
#    --num_gpus=0 \
#    --label_beta=0.95  \
#    --ae_vocab_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_vocab.in  \
#	--ae_emb_file=/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/copy_emb.json \
#    --load_model_cls=/Users/yxu132/pub-repos/tf-seq2seq/learned_models/def_output/nmt-T2.ckpt  \
#    --use_defending_as_target
