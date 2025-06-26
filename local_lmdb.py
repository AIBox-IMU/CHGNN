import lmdb
env = lmdb.open('GNN/../data/tcm_v2_ind/subgraphs_enclose_False_neg_1_hop_3', readonly=False, max_dbs=10)

# 打开子数据库
sub_db = env.open_db(b'sub_db_name')

with env.begin(write=True, db=sub_db) as txn:
    # 删除子数据库中的键
    txn.delete(b'your_key')

    # 或者清空子数据库
    txn.drop()