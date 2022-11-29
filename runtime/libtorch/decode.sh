export GLOG_logtostderr=1
export GLOG_v=2
wav_path=/home/work_nfs6/yzli/workshop/wenet_rnnt_runtime/example/aishell/rnnt/data/test_aishell/wav.scp
model_dir=/home/work_nfs6/yzli/workshop/wenet_rnnt_runtime/example/aishell/rnnt/exp/rnnt
./build/bin/decoder_main \
    --method "rnnt_greedy" \
    --chunk_size 16 \
    --num_left_chunks 10 \
    --wav_scp $wav_path \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/unit.txt 2>&1 | tee log_chunk16.txt
