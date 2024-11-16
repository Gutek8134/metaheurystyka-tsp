import random
from itertools import product


def random_instance(
    size: int,
    *,
    min_x: int = 1,
    max_x: int = 100000,
    min_y: int = 1,
    max_y: int = 100000,
) -> str:
    """
    :params:
    **min_x, max_x, min_y, max_y**: Determine range of x/y positions
    :raises ValueError:
    - instance size greater than max_x-min_x or max_y-min_y
    - maximal values lesser than minimal values
    """
    if size > (max_x - min_x + 1) * (max_y - min_y + 1):
        raise ValueError("Instance size too big")

    if max_x < min_x:
        raise ValueError("Maximum x must be greater or equal minimum x")
    if max_y < min_x:
        raise ValueError("Maximum y must be greater or equal minimum y")

    instance: str = str(size)
    values: list[tuple[int, int]] = [(random.randrange(
        min_x, max_x+1), random.randrange(min_y, max_y+1)) for _ in range(size)]

    for i, (position_x, position_y) in enumerate(values, start=1):
        instance += f"\n{i} {position_x} {position_y}"

    return instance
