import numpy as np
import time

def sign_op(my_train_list , de_w):          #
    sign_num = np.dot(my_train_list, de_w)
    if sign_num > 0:
        return 1
    else:
        return -1

def my_train(my_train_matrix,train_w,train_row):       #将经过处理的列表转换为各式矩阵
    insert_one = [1]* train_row
    train_y = my_train_matrix[:,-1]
    my_train_matrix=np.column_stack((insert_one, my_train_matrix))
    my_train_matrix=np.delete(my_train_matrix,train_col,axis = 1)
    np.savetxt(pathone, my_train_matrix, fmt="%f", delimiter=" ")
    dddi = 0
    de_count = 0
    while True:
        if dddi == train_row or de_count == 100:
            break
        de_sign = sign_op(my_train_matrix[dddi],train_w)
        if de_sign != train_y[dddi]:
            train_w = train_w + (train_y[dddi]) * my_train_matrix[dddi]
            de_count += 1
            dddi = -1
        dddi +=1
    return train_w

def my_test():
    my_test_matrix = np.loadtxt(pathtwo,delimiter=",")
    test_row = my_test_matrix.shape[0]
    test_col = my_test_matrix.shape[1]
    insert_one = np.ones((test_row))
    test_y = np.ones((test_row))
    answer_y = my_test_matrix[:,-1]
    my_test_matrix=np.column_stack((insert_one, my_test_matrix))
    my_test_matrix=np.delete(my_test_matrix,test_col,axis = 1)
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(0,test_row):
        de_sign = sign_op(my_test_matrix[i] , train_w)
        if de_sign > 0 and answer_y[i] > 0:
            TP += 1
        if de_sign > 0 and answer_y[i] < 0:
            FP += 1
        if de_sign < 0 and answer_y[i] > 0:
            FN += 1
        if de_sign < 0 and answer_y[i] < 0:
            TN += 1
    Accuracy = (TP+TN) / (TP+FP+TN+FN)
    if TP+FP == 0:
        Precision = 0
    else:
        Precision = (TP) / (TP+FP)
    if TP+FN == 0:
        Recall = 0
    else:
        Recall = (TP) / (TP+FN)
    if Precision+Recall == 0:
        F1 = 0
    else:
        F1 = (2*Precision*Recall)/(Precision+Recall)
    print(TP)
    print(FN)
    print(TN)
    print(FP)
    print("Accuracy:",Accuracy)
    print("Recall:", Recall)
    print("Precision:", Precision)
    print("F1:", F1)

if __name__ == '__main__':
    start_time = time.time()
    train_set = 'E:/B,B,B,BBox/大三上/人工智能/lab3数据/train.csv'
    pathtwo = 'E:/B,B,B,BBox/大三上/人工智能/lab3数据/val.csv'
    paththree = 'E:/B,B,B,BBox/大三上/人工智能/lab3数据/test.csv'
    pathone = 'E:/B,B,B,BBox/大三上/人工智能/lab3数据/one_hot.txt'
    my_train_matrix = np.loadtxt(train_set,delimiter=",")
    train_row = my_train_matrix.shape[0]
    train_col = my_train_matrix.shape[1]
    train_w = [1]* train_col
    train_w = my_train(my_train_matrix, train_w, train_row)
    print(train_w)
    my_test()
    end_time = time.time()
    print(end_time - start_time)
