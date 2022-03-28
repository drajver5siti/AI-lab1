#
#
# state = ((0,7), (0,8), (0,9))
#
# def parse(str):
#     x = int(str[0])
#     y = int(str[1])
#     return (x,y)
#
# # newHeadPos = (state[0][0], state[0][1] - 1)
# snakePos = list(state)
# while True:
#     inp = input()
#     if inp == "a":
#         snakePos.append((0,0))
#         continue
#     newHeadPos = parse(inp)
#     print(snakePos)
#     for i in range(len(snakePos) - 1, 0, -1):
#         snakePos[i] = (snakePos[i - 1][0], snakePos[i - 1][1])
#     snakePos[0] = newHeadPos
#     print(snakePos)
#
#
state = (((0, 7), (0, 8), (0, 9)), (2, 3))
print(state[0][0][0])