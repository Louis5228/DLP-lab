LOG=$(date +%m%d_%H%M_logs_task1)
echo $LOG
mkdir $LOG
python3 train.py --y_condition --output_dir $LOG \
                  --batch_size 8 \
                  --epochs 1500 \
                  --dataroot "/home/louis/DLP-lab/Lab7/dataset/task_1/" \
                  --dataset "task1" \
                  --classifier_weight "/home/louis/DLP-lab/Lab7/classifier_weight.pth" \
                  --flow_coupling "additive" \
                  --K 16 \
                  --L 4 \