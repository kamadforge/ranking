#reverses an array

combinationss = [

    [8, 4, 3, 9, 2, 1, 5, 6, 7, 0],
    [6, 16, 2, 19, 11, 3, 1, 18, 7, 15, 17, 8, 12, 14, 4, 9, 13, 0, 10, 5],
    [0, 30, 87, 66, 17, 28, 62, 54, 27, 11, 1, 86, 96, 93, 69, 48, 44, 84, 8, 41, 67, 47, 5, 53, 16, 97, 56, 23, 61, 18,
     24, 29, 43, 38, 59, 12, 72, 14, 85, 74, 21, 10, 83, 51, 26, 31, 35, 65, 6, 46, 99, 78, 42, 90, 49, 19, 55, 82, 22,
     91, 34, 36, 52, 81, 73, 98, 7, 2, 9, 58, 57, 95, 15, 71, 4, 75],
    [16, 10, 22, 21, 23, 11, 15, 18, 17, 20, 12, 24, 14, 13, 19, 6, 0, 4, 2, 3, 7, 9, 8, 1, 5]

]

new_combinations=[]

for i in combinationss:
    #print(i)
    new_combination=[]
    for el in reversed(i):
        new_combination.append(el)
    new_combinations.append(new_combination)

print(new_combinations)
