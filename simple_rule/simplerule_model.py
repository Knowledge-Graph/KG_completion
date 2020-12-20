import numpy as np
import scipy
import scipy.sparse as sp
import cPickle

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path, 'rb')))


def simplerule2(dataset,relid1,relid2):
    path = '../data/{}/'.format(dataset)
    # train
    inpl = load_file(path + '%s-train-lhs.pkl' % dataset)
    inpr = load_file(path + '%s-train-rhs.pkl' % dataset)
    inpo = load_file(path + '%s-train-rel.pkl' % dataset)

    # valid
    inpl_valid = load_file(path + '%s-valid-lhs.pkl' % dataset)
    inpr_valid = load_file(path + '%s-valid-rhs.pkl' % dataset)
    inpo_valid = load_file(path + '%s-valid-rel.pkl' % dataset)
    # test
    inpl_test = load_file(path + '%s-test-lhs.pkl' % dataset)
    inpr_test = load_file(path + '%s-test-rhs.pkl' % dataset)
    inpo_test = load_file(path + '%s-test-rel.pkl' % dataset)
    with open('./%s_rev_valid_train_80.pkl' % dataset, 'rb') as f:
        rel2rev = cPickle.load(f)
    with open('./%s_same_valid_train_80.pkl' % dataset, 'rb') as f:
        rel2same = cPickle.load(f)
    rev_rule=0
    notest=0
    for rid in range(relid1,relid2):
        if np.nonzero(inpo_test[rid])[1].any() == True:
            notest += len(np.nonzero(inpo_test[rid])[1])
            #print rid
            found=False
            if rid in rel2rev or rid in rel2same:


                # Test
                rel_row,rel_col=np.nonzero(inpo_test[rid])# indexes of test cases that contain 'test_rel'
                row,col=np.nonzero(inpl_test[:,rel_col])
                test_heads=row[np.argsort(col)] #row numbers of heads
                row,col=np.nonzero(inpr_test[:,rel_col])
                test_tails=row[np.argsort(col)] #row number of cols
                test_pairs=[i for i in zip(test_heads,test_tails)]
                for t in test_pairs:
                    if rid in rel2rev:
                        revid = rel2rev[rid]
                        head_row, head_col = np.nonzero(inpr[t[0]])
                        tail_row, tail_col = np.nonzero(inpl[t[1]])
                        rel_row, rel_col = np.nonzero(inpo[revid])

                        head_row, head_col2 = np.nonzero(inpr_valid[t[0]])
                        tail_row, tail_col2 = np.nonzero(inpl_valid[t[1]])
                        rel_row, rel_col2 = np.nonzero(inpo_valid[revid])
                        if len(set(head_col) & set(tail_col) & set(rel_col)) !=0 or len(set(head_col2) & set(tail_col2) & set(rel_col2)) !=0:
                            rev_rule+=1
                            found=True
                    if not found and rid in rel2same:
                        sameid = rel2same[rid]
                        head_row, head_col = np.nonzero(inpr[t[1]])
                        tail_row, tail_col = np.nonzero(inpl[t[0]])
                        rel_row, rel_col = np.nonzero(inpo[sameid])

                        head_row, head_col2 = np.nonzero(inpr_valid[t[0]])
                        tail_row, tail_col2 = np.nonzero(inpl_valid[t[1]])
                        rel_row, rel_col2 = np.nonzero(inpo_valid[sameid])

                        if len(set(head_col) & set(tail_col) & set(rel_col)) != 0 or len(
                                set(head_col2) & set(tail_col2) & set(rel_col2)) != 0:
                            rev_rule += 1





    print '{} result: {}'.format(dataset, rev_rule/float(notest)*100)


#
simplerule2('FB15k',14951, 16296)
simplerule2('FB15k-237',14505, 14742)
simplerule2('WN18',40943 ,40961)
simplerule2('WN18RR',40943, 40954)