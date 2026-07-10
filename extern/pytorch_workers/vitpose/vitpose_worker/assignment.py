from __future__ import annotations

import math


def hungarian_assign(costs: list[list[float]]) -> list[tuple[int, int]]:
    """Assign each object column to a distinct slot row with minimum cost."""
    if not costs or not costs[0]:
        return []
    slots = len(costs)
    objects = len(costs[0])
    if objects > slots:
        raise ValueError("cannot assign more objects than slots")

    memo: dict[tuple[int, int], tuple[float, list[int]]] = {}

    def solve(object_index: int, used_mask: int) -> tuple[float, list[int]]:
        key = (object_index, used_mask)
        if key in memo:
            return memo[key]
        if object_index == objects:
            return 0.0, []
        best_cost = math.inf
        best_slots: list[int] = []
        for slot_index in range(slots):
            if used_mask & (1 << slot_index):
                continue
            rest_cost, rest_slots = solve(
                object_index + 1,
                used_mask | (1 << slot_index),
            )
            total = float(costs[slot_index][object_index]) + rest_cost
            if total < best_cost:
                best_cost = total
                best_slots = [slot_index, *rest_slots]
        memo[key] = (best_cost, best_slots)
        return memo[key]

    _total, assigned_slots = solve(0, 0)
    return [(slot_index, object_index) for object_index, slot_index in enumerate(assigned_slots)]
