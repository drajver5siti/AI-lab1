

state = ((3,1), (2,1), (2, 0), (1, 0), (0,0))

state = list(state)
state.append((3, 3))
state = tuple(state)
print(state)

newHeadPos = (state[0][0], state[0][1] + 1)
snakePos = list(state)
print(snakePos)
for i in range(len(snakePos) - 1, 0, -1):
    snakePos[i] = (snakePos[i - 1][0], snakePos[i - 1][1])
snakePos[0] = newHeadPos


print(snakePos)
