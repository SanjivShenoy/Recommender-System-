import numpy as np
import random
import math
# from cur import cur_decomposition


def find_CUR (matrix, k):
    C = np.zeros([len(matrix), k])
    U = np.zeros([k, k])
    R = np.zeros([k, len(matrix[0])])
    
    row_prob = []
    col_prob = []
    mat_sum_sq = 0
    # getting row probabilities
    for i in range(len(matrix)):
        sum_sq = 0
        for j in range(len(matrix[0])):
            sum_sq += (matrix[i][j])**2
        mat_sum_sq += sum_sq
        row_prob.append(sum_sq)

    for i in range(len(row_prob)):
        row_prob[i] = row_prob[i]/mat_sum_sq
    
    # getting column probabilities
    for i in range(len(matrix[0])):
        sum_sq = 0
        for j in range(len(matrix)):
            sum_sq += (matrix[j][i])**2
        sum_sq = sum_sq/mat_sum_sq
        col_prob.append(sum_sq)

    row_choices = []
    col_choices = []
    # print(row_prob)
    # print(col_prob)
        
    # Forming matrix R
    for i in range(k):
        choose_row = np.random.choice(range(len(matrix)), p = row_prob)
        row_choices.append(choose_row)
        for j in range(len(matrix[0])):
            R[i][j] = matrix[choose_row][j]
            
    # Forming matrix C
    for i in range(k):
        choose_col = np.random.choice(range(len(matrix[0])), p = col_prob)
        col_choices.append(choose_col)
        for j in range(len(matrix)):
            C[j][i] = matrix[j][choose_col]
    """print("row choices...")
    print(row_coices)
    print("col choices...")
    print(col_choices)"""
    
        
    # Forming matrix U
    W = np.zeros([k, k])
    for i in range(len(row_choices)):
        for j in range(len(col_choices)):
            W[i][j] = matrix[row_choices[i]][col_choices[j]]
    X, sigma, Y_trans = np.linalg.svd(W, full_matrices=False)

    diag_sq = 0
    sq_90 = 0

    for i in range(len(sigma)):
        diag_sq += sigma[i]*sigma[i]

    for i in range(len(sigma)):
        sq_90 += sigma[i]*sigma[i]
        if(sq_90 > 0.9*diag_sq):
            num_val_retained = i+1
            break

    # print(sigma.shape)

    sigma = sigma[0:num_val_retained]
    X = X[:,0:num_val_retained]
    Y_trans = Y_trans[0:num_val_retained,:]

    # print(sigma.shape)

    #print(type(X))
    #print(type(sigma))
    
    for i in range(len(sigma)):
        if sigma[i] != 0:
            sigma[i] = 1/sigma[i]
            
    pseudo_sigma = np.diag(sigma)
    #print(pseudo_sigma)
    Y = Y_trans.transpose()
    temp = np.dot(Y, pseudo_sigma)
    temp = np.dot(temp, pseudo_sigma)
    temp = np.dot(temp, X.transpose())
    U = temp

    return (C, U, R)


if __name__ == '__main__':

    f=open("ratings.dat", "r")
    ratings = []
    if f.mode == 'r':
        contents = f.read()
        ratings = contents.strip().split('\n')
    f.close()

    ratings_list =[]
    for r in ratings:
        r = r.split('::')
        ratings_list.append([int(r[0]), int(r[1]), int(r[2]), int(r[3])])

    print("ratings done")

    random.shuffle(ratings_list)
    test_data = ratings_list[int(0.7*len(ratings_list)):]
    train_data =  ratings_list[:int(0.7*len(ratings_list))]

    NO_USERS = 6041 #Max of userID
    NO_MOVIES = 3953 #max of movieID
    matrix = np.zeros((NO_MOVIES , NO_USERS))
    matrix2 = np.zeros((NO_MOVIES , NO_USERS))

    for r in ratings_list:
        matrix[r[1]][r[0]] = r[2]

    C, U, R = find_CUR(matrix,2)

    print(C.shape, U.shape, R.shape)

    temp = np.dot(U,R)
    matrix_new_cur = np.dot(C,temp)

    error_cur = (matrix - matrix_new_cur)

    sum2 = 0
    for i in range(len(error_cur)):
        for j in range(len(error_cur[i])):
            sum2+=(error_cur[i][j]*error_cur[i][j])

    sum2/=(NO_USERS*NO_MOVIES)

    rmse_cur = math.sqrt(sum2)

    print(rmse_cur)
