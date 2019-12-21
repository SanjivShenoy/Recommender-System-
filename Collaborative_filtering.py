# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:13:22 2019

@author: Manish
"""
import numpy as np
import random
import math
from timeit import default_timer as timer

if __name__ == '__main__':
    f=open("movies.dat", "r")
    
    movies = []
    if f.mode == 'r':
        contents = f.read()
        movies = contents.strip().split('\n')
    
    f.close()
    
    movies_list = []
    for m in movies:
        m = m.split('::')
        m2 = m[2].split('|')
        movies_list.append([int(m[0]), m[1], m2])
    
    genres_dict = {}
    for movie in movies_list:
        for genre in movie[2]:
            if genre in genres_dict:
                genres_dict[genre].append(movie[0])
            else:
                genres_dict[genre] = [movie[0]]
    print("movies_list done")
    #print(genres_dict)
    
    f=open("users.dat", "r")
    users = []
    if f.mode == 'r':
        contents = f.read()
        users = contents.strip().split('\n')
  
    f.close()
    
    users_list = []
    for u in users:
        u = u.split('::')
        users_list.append([int(u[0]), u[1], int(u[2]), int(u[3]), u[4]])
        
    print("users_list done")
    
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
    
    avg = np.zeros(NO_MOVIES)
    for row in range(0,NO_MOVIES):
        avg[row] = np.average(matrix[row])
        for col in range(0, NO_USERS):
            if matrix[row][col] != 0:
                matrix2[row][col] = matrix[row][col] - avg[row]
    
    sim = np.zeros((NO_MOVIES , NO_MOVIES))
    
    print("sim begin")
    start = timer()
    for row1 in range(0,NO_MOVIES):
        for  row2 in range(0,NO_MOVIES):
            sim[row1][row2] = np.dot(matrix2[row1], matrix2[row2])
    
    end = timer()
    print("sim done, time = ", -start + end)
    test_predn = []
    
    mae = 0
    rmse = 0
    
    print(sim[1][1], sim[2][2], sim[1][2])
    
    print("prediction start")
    start = timer()
    for t in test_data:
        row = t[1]
        rowsum = 0
        rating = 0
        for col,val in enumerate(sim[row]):
            if col != row:
                if val > 0:
                    if matrix[row][col] != 0:
                        rowsum += val
                        rating += val*matrix[row][col]
        if rowsum != 0:
            rating /= rowsum
        #print(rating, t[2])
        error = abs(t[2] - rating)
        mae += error
        rmse += error*error
        
    end = timer()
    print("prediction end, time taken = ", end - start)
    
    print(mae)
    mae /= len(test_data)
    print(mae)
    rmse1 = math.sqrt(rmse)
    print(rmse1)
    rmse2 = rmse/len(test_data)
    rmse2 = math.sqrt(rmse2)
    print(rmse2)
    
    