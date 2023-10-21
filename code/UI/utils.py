def findMin(numbers:list):
    idx = 0
    for i, num in enumerate(numbers):
        if num < numbers[idx]:
            idx = i
    return idx

def pointDist(point1:tuple, point2:tuple):
    diff_x = point2[0] - point1[0]
    diff_y = point2[1] - point1[1]

    dist = (diff_x ** 2 + diff_y ** 2) ** 0.5
    return dist

def findMax(numbers:list):
    idx = 0
    for i, num in enumerate(numbers):
        if num > numbers[idx]:
            idx = i
    return idx

def getResult(op, operand1, operand2):
    if op == 1:
        return operand1 + operand2
    elif op == 2:
        return operand1 - operand2
    elif op == 3:
        return operand1 * operand2
    elif operand2 == 0:
        return None
    else:
        return operand1 / operand2