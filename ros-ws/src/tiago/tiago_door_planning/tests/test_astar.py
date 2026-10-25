from tiago_door_planning.astar import astar

WALL = {(1, 0), (1, 1), (1, 2)}


def grid_successors(state, data):
    x, y = state
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (x + dx, y + dy)
        if nxt not in WALL and -3 <= nxt[0] <= 3 and -3 <= nxt[1] <= 3:
            yield nxt, 1.0, data + 1


def manhattan_to(goal):
    return lambda state: abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def test_finds_shortest_path_around_wall():
    goal = (2, 0)

    path, data, _ = astar((0, 0), 0, lambda s: s == goal, grid_successors, manhattan_to(goal), 1000)

    assert path[0] == (0, 0) and path[-1] == goal
    assert len(path) - 1 == 4
    assert data[goal] == 4


def test_returns_none_when_goal_is_unreachable():
    path, _, _ = astar((0, 0), 0, lambda s: s == (9, 9), grid_successors, manhattan_to((9, 9)), 1000)

    assert path is None
