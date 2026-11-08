import heapq
import itertools


def reconstruct_path(parents, state):
    path = [state]
    while state in parents:
        state = parents[state]
        path.append(state)
    return path[::-1]


def astar(
    start,
    start_data,
    is_goal,
    successors,
    heuristic,
    max_expansions,
    progress=None,
    stall_limit=None,
):
    """Return (path, data, expansions).

    successors(state, data, may_improve) yields (next_state, cost, next_data) triples; may_improve(next_state,
    lower_bound) tells whether an edge costing at least lower_bound could still be kept, so successors can skip
    computing edges that cannot. Without a goal the path is None, or,
    when progress is given, the path to the expanded state with the highest progress (the earliest on ties).
    With progress, the search also gives up after stall_limit expansions without new progress.
    """
    tie_breaker = itertools.count()
    open_heap = [(heuristic(start), 0.0, next(tie_breaker), start)]
    costs = {start: 0.0}
    data = {start: start_data}
    parents = {}
    closed = set()
    best, best_expansion = start, 0
    while open_heap and len(closed) < max_expansions:
        _, _, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        if is_goal(state):
            return reconstruct_path(parents, state), data, len(closed)
        closed.add(state)
        if progress is not None and progress(state) > progress(best):
            best, best_expansion = state, len(closed)
        if stall_limit is not None and len(closed) - best_expansion > stall_limit:
            break
        state_cost = costs[state]

        def may_improve(next_state, lower_bound):
            return (
                next_state not in costs or state_cost + lower_bound < costs[next_state]
            )

        for next_state, step_cost, next_data in successors(
            state, data[state], may_improve
        ):
            cost = state_cost + step_cost
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
    if progress is None:
        return None, data, len(closed)
    return reconstruct_path(parents, best), data, len(closed)
