export CUDA_VISIBLE_DEVICES=0

for lr in "0.000005" "0.00001" "0.00005" "0.0001"
do
	save_lr_dir=`echo $lr | cut -d'.' -f 2`

	for bsz in 8 32
	do
		save_lr_dir=`echo $lr | cut -d'.' -f 2`
		python fine_tune.py \
		--data_dir './data' \
		--output_dir './results/lr_'${save_lr_dir}"_bsz_"${bsz}"_wd_0" \
		--lr ${lr} \
		--bsz ${bsz} \
		--wd 0
	done
done