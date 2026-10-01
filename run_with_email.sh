method=colmap
scene=glass_house

baselines/$method/run_container.sh python run_sequence.py \
    --data ../configs/data_example/$scene.yaml --model ../configs/$method.yaml

echo -e "Subject: $method $scene Done \n\nDone on $HOSTNAME " | sendmail james.gray1@sydney.edu.au
