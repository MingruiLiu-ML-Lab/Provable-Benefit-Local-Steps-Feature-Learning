dset="FeatureCIFAR"
name="final"
lr=0.01
save_idx=97
seeds=3
bs=64
feature_noise=0

augments=("1" "0")
lrs=("0.01" "0.03")
RIs=("16384" "65536")

Hs=("0.25" "0.5")

Is=("1024" "256" "64" "32" "16" "8")

# Parallel SGD
total_bs=$(($bs*8))
t=0
for (( j=0; j<$seeds; j++ )); do
    for (( i=0; i<${#augments[@]}; i++ )); do
        augment=${augments[i]}
        lr=${lrs[i]}
        RI=${RIs[i]}

        H=0.0
        dummy_H=${Hs[0]}
        I=128
        R=$((RI/I))

        bash run.sh ${save_idx}_${dset}_${name}/augment_${augment}_H_${dummy_H}/seed_${j} parallel_sgd $dset fedavg 1 1 $R $I $H $lr 1e8 1 0 1 $t $j $total_bs $augment $feature_noise &
        t=$((t+1))
    done
done
wait

# Local SGD
for (( l=0; l<$seeds; l++ )); do

    for (( i=0; i<${#augments[@]}; i++ )); do
        augment=${augments[i]}
        lr=${lrs[i]}
        RI=${RIs[i]}

        for (( j=0; j<${#Hs[@]}; j++ )); do
            H=${Hs[j]}

            for (( k=0; k<${#Is[@]}; k++ )); do
                I=${Is[k]}
                R=$((RI/I))

                bash run.sh ${save_idx}_${dset}_${name}/augment_${augment}_H_${H}/seed_${l} local_sgd_I_${I}_R_${R} $dset fedavg 8 8 $R $I $H $lr 1e8 1 0 8 0 $l $bs $augment $feature_noise
            done
        done
    done
done
