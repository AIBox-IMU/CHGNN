## An example for training
#python GNN/train.py --gpu 0 -d WN18RR_v1 -e test --batch_size 512 -dim 32 --dropout 0 -l 6 --num_epochs 10
#
## To train with nohup
#nohup GNN/train.py --gpu 0 -d WN18RR_v1 -e test --batch_size 512 -dim 32 --dropout 0 -l 6 --num_epochs 10 > /dev/null 2>&1 &
#
## To test the model
#python GNN/test_ranking.py --gpu 0 -d WN18RR_v1_ind -e test
python GNN/train.py --gpu 0 -d WN18RR_v1 -e hpy_wn_v1 --batch_size 512 -dim 32 --dropout 0 -l 6 --num_epochs 1
python GNN/train.py --gpu 0 -d tcm_v2 -e tcm_v2_att_loss --batch_size 64  -dim 32 --dropout 0 -l 6 --num_epochs 10 --loss fl

