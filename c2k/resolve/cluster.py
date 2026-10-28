"""Union-find over accepted pairs -> entities (§6.4). Canonical name = the
most frequent surface form in the cluster, ties broken by longest string."""

from __future__ import annotations

from collections import Counter


class UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def cluster_mentions(n_mentions: int, accepted_pairs: list[tuple[int, int]]) -> dict[int, int]:
    """Returns mention_index -> cluster_root_index for all `n_mentions`
    (singletons included, mapped to themselves)."""
    uf = UnionFind(n_mentions)
    for i, j in accepted_pairs:
        uf.union(i, j)
    return {i: uf.find(i) for i in range(n_mentions)}


def canonical_names(surfaces: list[str], clusters: dict[int, int]) -> dict[int, str]:
    """cluster_root_index -> canonical surface name."""
    groups: dict[int, list[str]] = {}
    for i, root in clusters.items():
        groups.setdefault(root, []).append(surfaces[i])

    result: dict[int, str] = {}
    for root, group_surfaces in groups.items():
        counts = Counter(group_surfaces)
        max_count = max(counts.values())
        tied = [s for s, c in counts.items() if c == max_count]
        result[root] = max(tied, key=len)
    return result
