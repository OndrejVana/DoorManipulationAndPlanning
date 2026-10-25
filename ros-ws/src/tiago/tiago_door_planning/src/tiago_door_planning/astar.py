import heapq
import itertools


def reconstruct_path(parents, state):
    path = [state]
    while state in parents:
        state = parents[state]
        path.append(state)
    return path[::-1]


def astar(start, start_data, is_goal, successors, heuristic, max_expansions):
    """Return (path, data, expansions); path is None when no goal is found.

    successors(state, data) yields (next_state, cost, next_data) triples.
    """
    tie_breaker = itertools.count()
    open_heap = [(heuristic(start), 0.0, next(tie_breaker), start)]
    costs = {start: 0.0}
    data = {start: start_data}
    parents = {}
    closed = set()
    while open_heap and len(closed) < max_expansions:
        _, _, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        if is_goal(state):
            return reconstruct_path(parents, state), data, len(closed)
        closed.add(state)
        for next_state, step_cost, next_data in successors(state, data[state]):
            cost = costs[state] + step_cost
            if next_state not in costs or cost < costs[next_state]:
                costs[next_state] = cost
                data[next_state] = next_data
                parents[next_state] = state
                heapq.heappush(
                    open_heap,
                    (
                        cost + heuristic(next_state),
                        -cost,
                        next(tie_breaker),
                        next_state,
                    ),
                )
    return None, data, len(closed)
