#失败，因为有的输入不可被哈希，故无法作为字典的键
from functools import wraps
functionResultMap = {}
def quick_fun(func):
    #装饰器，如果函数已经输入过一次就快速返回存储好的结果
    @wraps(func)
    def wrapper(*args,**kw):
        temp = args
        funcunit = functionResultMap.get(func.__name__)
        if not funcunit:
            functionResultMap[func.__name__] = {}
            result = functionResultMap[func.__name__][temp] = func(*args,**kw)
        else:
            result = funcunit.get(temp)
            if not result:
                result = functionResultMap[func.__name__][temp] = func(*args,**kw)
        return result
    return wrapper

def unit_test():
    @quick_fun
    def addone(x,y):
        return x+1
    print(addone(0,3))
    print(addone(1,1))
    print(addone(2,5))
    print(addone(1,6))
    print(addone(2,5))