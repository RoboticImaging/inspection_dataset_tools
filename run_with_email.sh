method=colmap
scene=office_dynamic

# baselines/$method/run_container.sh python run_sequence.py \
#     --data ../configs/data_example/$scene.yaml --model ../configs/$method.yaml
CUDA_VISIBLE_DEVICES=0 baselines/$method/run_container.sh python run_sequence.py \
    --data ../configs/data_office/office_night.yaml --model ../configs/$method.yaml &
CUDA_VISIBLE_DEVICES=1 baselines/$method/run_container.sh python run_sequence.py \
    --data ../configs/data_office/office_overcast.yaml --model ../configs/$method.yaml
wait
# baselines/$method/run_container.sh python run_sequence.py \
#     --data ../configs/data_office/office_overcast.yaml --model ../configs/$method.yaml &
# baselines/$method/run_container.sh python run_sequence.py \
#     --data ../configs/data_office/office_sunrise.yaml --model ../configs/$method.yaml &
# wait

echo -e "Subject: $method $scene Done \n\nDone on $HOSTNAME for overcast and night" | sendmail james.gray1@sydney.edu.au
