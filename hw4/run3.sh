export CUDA_VISIBLE_DEVICES=0

for lr in "0.0001"
do
	save_lr_dir=`echo $lr | cut -d'.' -f 2`

	for bsz in 32
	do
		python fine_tune.py \
		--data_dir './data' \
		--output_dir './results/lr_'${save_lr_dir}"_bsz_"${bsz}"_wd_01" \
		--lr ${lr} \
		--bsz ${bsz} \
		--wd 0.01
	done
done