import bisect


"""
Дефинирање на класа за структурата на проблемот кој ќе го решаваме со пребарување.
Класата Problem е апстрактна класа од која правиме наследување за дефинирање на основните 
карактеристики на секој проблем што сакаме да го решиме
"""


class Problem:
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """За дадена состојба, врати речник од парови {акција : состојба}
        достапни од оваа состојба. Ако има многу следбеници, употребете
        итератор кој би ги генерирал следбениците еден по еден, наместо да
        ги генерирате сите одеднаш.
        :param state: дадена состојба
        :return:  речник од парови {акција : состојба} достапни од оваа
                  состојба
        :rtype: dict
        """
        raise NotImplementedError

    def actions(self, state):
        """За дадена состојба state, врати листа од сите акции што може да
        се применат над таа состојба
        :param state: дадена состојба
        :return: листа на акции
        :rtype: list
        """
        raise NotImplementedError

    def result(self, state, action):
        """За дадена состојба state и акција action, врати ја состојбата
        што се добива со примена на акцијата над состојбата
        :param state: дадена состојба
        :param action: дадена акција
        :return: резултантна состојба
        """
        raise NotImplementedError

    def goal_test(self, state):
        """Врати True ако state е целна состојба. Даденава имплементација
        на методот директно ја споредува state со self.goal, како што е
        специфицирана во конструкторот. Имплементирајте го овој метод ако
        проверката со една целна состојба self.goal не е доволна.
        :param state: дадена состојба
        :return: дали дадената состојба е целна состојба
        :rtype: bool
        """
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Врати ја цената на решавачкиот пат кој пристигнува во состојбата
        state2 од состојбата state1 преку акцијата action, претпоставувајќи
        дека цената на патот до состојбата state1 е c. Ако проблемот е таков
        што патот не е важен, оваа функција ќе ја разгледува само состојбата
        state2. Ако патот е важен, ќе ја разгледува цената c и можеби и
        state1 и action. Даденава имплементација му доделува цена 1 на секој
        чекор од патот.
        :param c: цена на патот до состојбата state1
        :param state1: дадена моментална состојба
        :param action: акција која треба да се изврши
        :param state2: состојба во која треба да се стигне
        :return: цена на патот по извршување на акцијата
        :rtype: float
        """
        return c + 1

    def value(self):
        """За проблеми на оптимизација, секоја состојба си има вредност. 
        Hill-climbing и сличните алгоритми се обидуваат да ја максимизираат
        оваа вредност.
        :return: вредност на состојба
        :rtype: float
        """
        raise NotImplementedError


"""
Дефинирање на класата за структурата на јазел од пребарување.
Класата Node не се наследува
"""


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Креирај јазол од пребарувачкото дрво, добиен од parent со примена
        на акцијата action
        :param state: моментална состојба (current state)
        :param parent: родителска состојба (parent state)
        :param action: акција (action)
        :param path_cost: цена на патот (path cost)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0  # search depth
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """Излистај ги јазлите достапни во еден чекор од овој јазол.
        :param problem: даден проблем
        :return: листа на достапни јазли во еден чекор
        :rtype: list(Node)
        """

        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """Дете јазел
        :param problem: даден проблем
        :param action: дадена акција
        :return: достапен јазел според дадената акција
        :rtype: Node
        """
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))

    def solution(self):
        """Врати ја секвенцата од акции за да се стигне од коренот до овој јазол.
        :return: секвенцата од акции
        :rtype: list
        """
        return [node.action for node in self.path()[1:]]

    def solve(self):
        """Врати ја секвенцата од состојби за да се стигне од коренот до овој јазол.
        :return: листа од состојби
        :rtype: list
        """
        return [node.state for node in self.path()[0:]]

    def path(self):
        """Врати ја листата од јазли што го формираат патот од коренот до овој јазол.
        :return: листа од јазли од патот
        :rtype: list(Node)
        """
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        result.reverse()
        return result

    """Сакаме редицата од јазли кај breadth_first_search или 
    astar_search да не содржи состојби - дупликати, па јазлите што
    содржат иста состојба ги третираме како исти. [Проблем: ова може
    да не биде пожелно во други ситуации.]"""

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


"""
Дефинирање на помошни структури за чување на листата на генерирани, но непроверени јазли
"""


class Queue:
    """Queue е апстрактна класа / интерфејс. Постојат 3 типа:
        Stack(): Last In First Out Queue (стек).
        FIFOQueue(): First In First Out Queue (редица).
        PriorityQueue(order, f): Queue во сортиран редослед (подразбирливо,од најмалиот кон
                                 најголемиот јазол).
    """

    def __init__(self):
        raise NotImplementedError

    def append(self, item):
        """Додади го елементот item во редицата
        :param item: даден елемент
        :return: None
        """
        raise NotImplementedError

    def extend(self, items):
        """Додади ги елементите items во редицата
        :param items: дадени елементи
        :return: None
        """
        raise NotImplementedError

    def pop(self):
        """Врати го првиот елемент од редицата
        :return: прв елемент
        """
        raise NotImplementedError

    def __len__(self):
        """Врати го бројот на елементи во редицата
        :return: број на елементи во редицата
        :rtype: int
        """
        raise NotImplementedError

    def __contains__(self, item):
        """Проверка дали редицата го содржи елементот item
        :param item: даден елемент
        :return: дали queue го содржи item
        :rtype: bool
        """
        raise NotImplementedError


class Stack(Queue):
    """Last-In-First-Out Queue."""

    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop()

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class FIFOQueue(Queue):
    """First-In-First-Out Queue."""

    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class PriorityQueue(Queue):
    """Редица во која прво се враќа минималниот (или максималниот) елемент
    (како што е определено со f и order). Оваа структура се користи кај
    информирано пребарување"""
    """"""

    def __init__(self, order=min, f=lambda x: x):
        """
        :param order: функција за подредување, ако order е min, се враќа елементот
                      со минимална f(x); ако order е max, тогаш се враќа елементот
                      со максимална f(x).
        :param f: функција f(x)
        """
        assert order in [min, max]
        self.data = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort_right(self.data, (self.f(item), item))

    def extend(self, items):
        for item in items:
            bisect.insort_right(self.data, (self.f(item), item))

    def pop(self):
        if self.order == min:
            return self.data.pop(0)[1]
        return self.data.pop()[1]

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.data)

    def __getitem__(self, key):
        for _, item in self.data:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.data):
            if item == key:
                self.data.pop(i)
                
from sys import maxsize as infinity

"""
Информирано пребарување во рамки на граф
"""


def memoize(fn, slot=None):
    """ Запамети ја пресметаната вредност за која била листа од
    аргументи. Ако е специфициран slot, зачувај го резултатот во
    тој slot на првиот аргумент. Ако slot е None, зачувај ги
    резултатите во речник.
    :param fn: зададена функција
    :type fn: function
    :param slot: име на атрибут во кој се чуваат резултатите од функцијата
    :type slot: str
    :return: функција со модификација за зачувување на резултатите
    :rtype: function
    """
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


def best_first_graph_search(problem, f):
    """Пребарувај низ следбениците на даден проблем за да најдеш цел. Користи
     функција за евалуација за да се одлучи кој е сосед најмногу ветува и
     потоа да се истражи. Ако до дадена состојба стигнат два пата, употреби
     го најдобриот пат.
    :param problem: даден проблем
    :type problem: Problem
    :param f: дадена функција за евалуација (проценка)
    :type f: function
    :return: Node or None
    :rtype: Node
    """
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def greedy_best_first_graph_search(problem, h=None):
    """ Greedy best-first пребарување се остварува ако се специфицира дека f(n) = h(n).
    :param problem: даден проблем
    :type problem: Problem
    :param h: дадена функција за хевристика
    :type h: function
    :return: Node or None
    """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar_search(problem, h=None):
    """ A* пребарување е best-first graph пребарување каде f(n) = g(n) + h(n).
    :param problem: даден проблем
    :type problem: Problem
    :param h: дадена функција за хевристика
    :type h: function
    :return: Node or None
    """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def recursive_best_first_search(problem, h=None):
    """Recursive best first search - ја ограничува рекурзијата
    преку следење на f-вредноста на најдобриот алтернативен пат
    од било кој јазел предок (еден чекор гледање нанапред).
    :param problem: даден проблем
    :type problem: Problem
    :param h: дадена функција за хевристика
    :type h: function
    :return: Node or None
    """
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0  # (втората вредност е неважна)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Подреди ги според најниската f вредност
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result
    


class Snake(Problem):

    def __init__(self, grid_size, initial, goal=None):
        super().__init__(initial, goal)
        self.gridSize = grid_size


    def goal_test(self, state):
        return state[0] == 0


    def moveSnake(self, snakePos, newHeadPos):
        snakePos = list(snakePos)
        for i in range(len(snakePos) - 1, 0, -1):
            snakePos[i] = (snakePos[i - 1][0], snakePos[i - 1][1])
        snakePos[0] = newHeadPos
        return snakePos



    def successor(self, state):
        succ = dict()
        # state[0] -> green_app_count
        # state[1] -> green_app_positions
        # state[2] -> DIRECTION
        # state[3] -> snakePos


        if state[4] == "SOUTH":
            # snakePos[0] -> head of snake

            newHeadPos = (state[5][0][0], state[5][0][1] - 1)
            # go forwards
            if newHeadPos[1] > 0 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    # grow the snake, the position of the added piece doesnt matter it will sort itself out on the next move
                    snakePos.append((0, 0))

                succ["ProdolzhiPravo"] = (newGreenAppCount, newGreenAppPos, "SOUTH", tuple(snakePos))


            newHeadPos = (state[5][0][0] - 1, state[5][0][1])
            # go right
            if newHeadPos[0] > 0 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount-= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))
                                
                succ["SvrtiDesno"] = (newGreenAppCount, newGreenAppPos, "WEST", tuple(snakePos))


            newHeadPos = (state[5][0][0] + 1, state[5][0][1])
            # go left
            if newHeadPos[0] < self.gridSize - 1 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))

                succ["SvrtiLevo"] = (newGreenAppCount, newGreenAppPos, "EAST", tuple(snakePos))


        elif state[4] == "NORTH":
            newHeadPos = (state[5][0][0], state[5][0][1] + 1)
            # go forwards
            if newHeadPos[1] < self.gridSize - 1 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)            
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))

                succ["ProdolzhiPravo"] = (newGreenAppCount, newGreenAppPos, "NORTH", tuple(snakePos))


            newHeadPos = (state[5][0][0] + 1, state[5][0][1])
            # go right
            if newHeadPos[0] < self.gridSize - 1 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))

                succ["SvrtiDesno"] = (newGreenAppCount, newGreenAppPos, "EAST", tuple(snakePos)) 


            newHeadPos = (state[5][0][0] - 1, state[5][0][1])
            if newHeadPos[0] > 0 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))

                succ["SvrtiLevo"] = (newGreenAppCount, newGreenAppPos, "WEST", tuple(snakePos)) 
                

        elif state[4] == "EAST":
            newHeadPos = (state[5][0][0] + 1, state[5][0][0])
            if newHeadPos[0] < self.gridSize - 1 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))

                succ["ProdolzhiPravo"] = (newGreenAppCount, newGreenAppPos, "EAST", tuple(snakePos)) 


            newHeadPos = (state[5][0][0], state[5][0][0] - 1)
            if newHeadPos[1] > 0 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))
 

                succ["SvrtiDesno"] = (newGreenAppCount, newGreenAppPos,  "SOUTH", tuple(snakePos)) 


            newHeadPos = (state[5][0][0], state[5][0][0] + 1)
            if newHeadPos[1] < self.gridSize - 1 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))
  

                succ["SvrtiLevo"] = (newGreenAppCount, newGreenAppPos, "NORTH", tuple(snakePos)) 


        elif state[4] == "WEST":
            newHeadPos = (state[5][0][0] - 1, state[5][0][0])
            if newHeadPos[0] > 0 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))


                succ["ProdolzhiPravo"] = (newGreenAppCount, newGreenAppPos, "WEST", tuple(snakePos)) 


            newHeadPos = (state[5][0][0], state[5][0][0] + 1)
            if newHeadPos[1] < self.gridSize - 1 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))
 

                succ["SvrtiDesno"] = (newGreenAppCount, newGreenAppPos, "NORTH", tuple(snakePos)) 


            newHeadPos = (state[5][0][0], state[5][0][0] - 1)
            if newHeadPos[1] > 0 and newHeadPos not in state[3]:
                snakePos = self.moveSnake(state[5], newHeadPos)
                newGreenAppCount = state[0]
                newGreenAppPos = state[1]

                if newHeadPos in state[1]:
                    newGreenAppCount -= 1
                    newGreenAppPos = tuple(filter(lambda x: x != newHeadPos, newGreenAppPos))
                    snakePos.append((0, 0))


                succ["SvrtiLevo"] = (newGreenAppCount, newGreenAppPos, "SOUTH", tuple(snakePos)) 
        
        return succ

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        # print(self.successor(state)[action][5][0])
        self.state = self.successor(state)[action]
        return self.successor(state)[action]

    def value(self, state):
        pass

    def h(self):
        # print(self.state)
        # val += mhd(state[3][0], state[1])
        # for x in state[1]:
            # val+= mhd(state[3][0], x)
        # return val
        pass

def mhd(headPos, otherPos):
    return abs(headPos[0] - otherPos[0]) + abs(headPos[1] - otherPos[1])

def parsePoints(point):
    tup = (point[0], point[2])
    return tup


if __name__ == '__main__':
    grid_size = 10
    green_apples_count = int(input())
    green_apples_positions = []
    for x in range(green_apples_count):
        green_apples_positions.append(parsePoints(input()))


    snakePos = ((0, 7), (0, 8), (0, 9))

    s = Snake(grid_size, (green_apples_count, tuple(green_apples_positions), "SOUTH", snakePos))

    print(astar_search(s))