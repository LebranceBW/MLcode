'''
西瓜数据集3.0阿尔法
x0 表示密度 x1表示含糖量 y = 1表示是好瓜，反之是烂瓜🍉
wm_dataset 格式(+1 为正例， 0为反例)
[[[x11,x12],y1],
[[x21,x22],y2],
]

'''
watermelon_counterexample_x = [
    [0.666, 0.091],
    [0.243, 0.267],
    [0.245, 0.057],
    [0.343, 0.099],
    [0.639, 0.161],
    [0.657, 0.198],
    [0.360, 0.370],
    [0.593, 0.042],
    [0.719, 0.103],
]

watermelon_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

watermelon_posiexam_x = [
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.318],
    [0.556, 0.215],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.437, 0.211],
]

watermelon_x = watermelon_posiexam_x+watermelon_counterexample_x

wm_dataSet = list(zip(watermelon_x, watermelon_y))
pass
