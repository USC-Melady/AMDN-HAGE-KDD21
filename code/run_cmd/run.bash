data_dir='kdd_data/'
data_prefix='eachseq_data_minlen2'
out_dir='./'

max_seq_length=128
encoder_type='ATTN'
history_size=32
mark_embedding_size=64

n_components=8
seed=1
batch_size=128

max_epochs=1000
patience=10
save_freq=10
gpu0sz=64

python ../amdn_Gaussian-tied-unco-new.py --data_dir=$data_dir --data_prefix=$data_prefix \
--out_dir=$out_dir --max_seq_length=$max_seq_length \
--encoder_type=$encoder_type --history_size=$history_size --mark_embedding_size=$mark_embedding_size  \
--n_components=$n_components --seed=$seed --batch_size $batch_size --max_epochs $max_epochs \
--patience $patience --save_freq $save_freq --gpu0sz $gpu0sz