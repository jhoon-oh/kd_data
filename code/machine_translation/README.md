# ReadMe

## Requirements

---

### Package Download

- [PyTorch](http://pytorch.org/) version >= 1.4.0
- Python version >= 3.6
- numpy

```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable ./
```

- Download distillation data set is provided by fairseq Team [[link](http://dl.fbaipublicfiles.com/nat/distill_dataset.zip)] in path fairseq/examples/translation/
- Make reduced data file. Reduced data will be located /examples/translation/wmt14_ende_distill_{ratio} respectively

    ratio = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

```bash
python prepare_reduce_data.py
```

- Data preprocessing

```bash
FILE_NAME=wmt14_ende_distill_0.9
TEXT=examples/translation/$FILE_NAME
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train.en-de --validpref $TEXT/valid.en-de --testpref $TEXT/test.en-de \
    --destdir data-bin/$FILE_NAME --workers 20 --joined-dictionary

```

## Training and Validation

---

- We mostly follow the setting of the [previous research](https://arxiv.org/abs/1911.02727)

```bash
# Make Checkpoint folder
cd checkpoints
mkdir at_base
mkdir nat_base
cd ../

cd generate_results
mkdir at_base
mkdir nat_base
cd ../

```

### Autoregressive Transformer model (AT)

- We used Transformer implemented in Fairseq
- Different size of data can be  used by change {FILE_NAME}. Below example is for the 0.9 ratio case
- AT_big can be trained by changing the --arch argument to **transformer_wmt_en_de_big**

```bash

# Select Data
FILE_NAME=wmt14_ende_distill_0.9
EXP_NAME=at_base

# Train
fairseq-train \
    data-bin/$FILE_NAME\
    --arch transformer_wmt_en_de_big --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --fp16\
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --max-tokens 8000 --max-epoch 70 \
    --dropout 0.3 --weight-decay 0.0001 --save-dir checkpoints/$EXP_NAME/$FILE_NAME\
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format 'simple' --log-interval 100 --tensorboard-logdir tensorboard_log/at/$FILE_NAME \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric | tee -a /home/ubuntu/KD/Data/at_logs/$FILE_NAME.txt

# Generate
fairseq-generate data-bin/$FILE_NAME \
    --path checkpoints/$EXP_NAME/$FILE_NAME/checkpoint_last.pt  \
    --batch-size 128 --beam 5 --remove-bpe \
    --results-path generate_results/$EXP_NAME/$FILE_NAME  

cd generate_results/$EXP_NAME/$FILE_NAME  
grep ^T generate-test.txt | cut -f2- > target.txt
grep ^H generate-test.txt | cut -f3- > hypotheses.txt

# BLEU score
cd ../..
fairseq-score --sys .generate_results/$EXP_NAME/hypotheses.txt \
--ref .generate_results/$EXP_NAME/target.txt
```

### Non Autoregressive Transformer model (NAT)

- We used vanilla [NAT](https://arxiv.org/abs/1711.02281) for our experiment

```bash
# Select Data
FILE_NAME=wmt14_ende_distill_0.9
EXP_NAME=nat_base

# Train
fairseq-train \
    data-bin/$FILE_NAME\
    --save-dir checkpoints/$EXP_NAME/$FILE_NAME \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion nat_loss \
    --arch nonautoregressive_transformer \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 --fp16 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 --save-interval 10 --tensorboard-logdir tensorboard_log/$FILE_NAME \
    --fixed-validation-seed 7 \
    --max-tokens 8000 --max-epoch 70\
    --max-update 500000  \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples 

# Generate
fairseq-generate \
    data-bin/$FILE_NAME\
    --gen-subset test \
    --task translation_lev \
    --path checkpoints/$EXP_NAME/$FILE_NAME/checkpoint_last.pt \
--iter-decode-max-iter 0 \
    --results-path generate_results/$EXP_NAME/$FILE_NAME \
    --beam 1 --remove-bpe \
    --print-step --eval-bleu-print-samples \
    --batch-size 400 

cd generate_results/$FILE_NAME
grep ^T generate-test.txt | cut -f2- > target.txt
grep ^H generate-test.txt | cut -f3- > hypotheses.txt

# Score
cd ../..
fairseq-score --sys /generate_results/$FILE_NAME/hypotheses.txt \
--ref /home/ub
```
