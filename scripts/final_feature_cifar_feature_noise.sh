dset="FeatureCIFAR"
name="noise"
save_idx=101
seeds=3
bs=64
augment=0
lr=0.03
RI="98304"
I="32"
H="0.5"
gpus=8

feature_noises=("0.03125" "0.0625" "0.125" "0.25")

# Parallel SGD
total_bs=$(($bs*8))
t=0
for (( j=0; j<$seeds; j++ )); do
    for (( i=0; i<${#feature_noises[@]}; i++ )); do
        feature_noise=${feature_noises[i]}

        H=0.0
        dummy_H=${Hs[0]}
        I=128
        R=$((RI/I))

        bash run.sh ${save_idx}_${dset}_${name}/noise_${feature_noise}/seed_${j} parallel_sgd $dset fedavg 1 1 $R $I $H $lr 1e8 1 0 1 $t $j $total_bs $augment $feature_noise &
        t=$((t+1))
        if [ $t -ge $gpus ]; then
            wait
            t=0
        fi
    done
done
wait

# Local SGD
for (( l=0; l<$seeds; l++ )); do
    for (( i=0; i<${#feature_noises[@]}; i++ )); do
        feature_noise=${feature_noises[i]}
        R=$((RI/I))
        bash run.sh ${save_idx}_${dset}_${name}/noise_${feature_noise}/seed_${l} local_sgd_I_${I}_R_${R} $dset fedavg 8 8 $R $I $H $lr 1e8 1 0 8 0 $l $bs $augment $feature_noise
    done
done
