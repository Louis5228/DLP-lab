LOG=$(date +%m%d_%H%M_logs)
echo $LOG
mkdir $LOG
python3 train.py  --y_condition --output_dir $LOG \
                  --batch_size 16 \
                  --epochs 50 \
                  --dataroot "/home/louis/DLP-lab/Lab7/dataset/task_2/" \
                  --K 6 \
                  --L 3 \