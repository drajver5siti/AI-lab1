snakePos = [(0, 7), (0, 8), (0, 9)]

print(snakePos)

newHeadPos = (snakePos[0][0] + 1, snakePos[0][1])

for i in range(len(snakePos) - 1, 0, -1):
    snakePos[i] = (snakePos[i - 1][0], snakePos[i - 1][1])
snakePos[0] = newHeadPos


print(snakePos)