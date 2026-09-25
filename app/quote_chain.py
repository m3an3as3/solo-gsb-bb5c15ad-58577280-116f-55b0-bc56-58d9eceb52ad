"""多段引文的链式定位：在拼接文本中为 2～4 段按序引文选互不重叠的完整链。

提交与 :func:`app.assembler.assemble` 相同的有序 segments 与 2～4 段按原文
顺序排列的引文（各 1～80 字），先复用其拼接得到拼接文本与逐字来源（沿用段数/
长度限制与续接失败规则），再对**每段引文独立**枚举编辑代价不超过
:data:`~app.locator.MAX_COST` 的全部非空区间候选（相同字或任一侧 ``?`` 代价
0，替换、插入、删除各代价 1；即便某段单独最低代价的区间无法接入完整链，也要
保留该段其余合格候选），随后：

- 要求各引文选中的区间在拼接文本中**按序且互不重叠**
  （前一引文区间终点不超过后一引文区间起点，可首尾相接）；
- 以各段引文与区间代价之和为总代价，选总代价最低的**完整定位链**；
- 唯一最优链：返回各引文区间、片段、代价与该区间逐字 sources；
- 最优链由多条**不同区间序列**并列取得：抛出 :class:`QuoteChainTieError`，
  携带按区间序列字典序排列的前两条不同最优链；
- 不存在覆盖全部引文的完整链：抛出 :class:`QuoteChainNotFoundError`。

本模块只做定位，绝不改写拼接文本。
"""

from __future__ import annotations

from .assembler import assemble
from .locator import MAX_COST, MAX_QUOTE_LEN, enumerate_qualified_hits

MIN_QUOTES = 2   # 引文段数下限
MAX_QUOTES = 4   # 引文段数上限

# 每个 DP 状态至多保留的不同链数：最终只需字典序最小的前两条并列链。
_KEEP_CHAINS = 2


class QuoteChainNotFoundError(ValueError):
    """不存在覆盖全部引文、区间按序且互不重叠的完整定位链。"""


class QuoteChainTieError(ValueError):
    """总代价最低的完整定位链由多条不同区间序列并列取得。"""

    def __init__(self, cost: int, ties: list[dict]):
        self.cost = cost
        self.ties = ties
        super().__init__(
            f"总代价 {cost} 的完整定位链有多个并列，无法唯一定位"
        )


def _chain_key(intervals: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """区间序列的字典序键（逐 ``(start, end)`` 比较）。"""
    return tuple(intervals)


def _merge_chains(
    states: dict[int, list],
    end: int,
    cost: int,
    chains: list[list[tuple[int, int]]],
) -> None:
    """把以 ``end`` 结尾、总代价 ``cost`` 的若干链并入 DP 状态。

    每个终点只保留当前最低代价；同代价时只保留字典序最小的
    :data:`_KEEP_CHAINS` 条不同链。
    """
    kept = states.get(end)
    if kept is not None and cost > kept[0]:
        return
    if kept is None or cost < kept[0]:
        kept = [cost, []]
        states[end] = kept
    bucket: list[list[tuple[int, int]]] = kept[1]
    for chain in chains:
        if any(chain == other for other in bucket):
            continue
        bucket.append(chain)
        bucket.sort(key=_chain_key)
        if len(bucket) > _KEEP_CHAINS:
            bucket.pop()


def locate_quotes(segments: list[str], quotes: list[str]) -> dict:
    """在 2～5 段转写的拼接文本中为多段引文做链式定位。

    返回唯一最优链时的 ``{"text", "quotes", "cost", "hits"}``，其中
    ``hits[i]`` 为 ``{"quote", "start", "end", "fragment", "cost",
    "sources"}``；拼接阶段失败原样抛出
    :class:`~app.assembler.AssemblyError`，无完整链抛出
    :class:`QuoteChainNotFoundError`，多链并列抛出
    :class:`QuoteChainTieError`。
    """
    if not MIN_QUOTES <= len(quotes) <= MAX_QUOTES:
        raise ValueError(f"引文段数需为 {MIN_QUOTES}～{MAX_QUOTES} 段")
    for idx, quote in enumerate(quotes):
        if not 1 <= len(quote) <= MAX_QUOTE_LEN:
            raise ValueError(
                f"第 {idx + 1} 段引文长度需为 1～{MAX_QUOTE_LEN} 字"
            )

    result = assemble(segments)
    text, sources = result["text"], result["sources"]

    # 每段引文的全部合格候选 [(start, end, cost)]（按 (start, end) 升序）。
    candidate_lists: list[list[tuple[int, int, int]]] = []
    for qi, quote in enumerate(quotes):
        _best, candidates = enumerate_qualified_hits(text, quote, MAX_COST)
        if not candidates:
            raise QuoteChainNotFoundError(
                f"第 {qi + 1} 段引文（{len(quote)} 字）在拼接文本中不存在"
                f"编辑代价不超过 {MAX_COST} 的非空区间，无法构成完整定位链"
            )
        candidate_lists.append(candidates)

    def make_hit(qi: int, start: int, end: int, cost: int) -> dict:
        return {
            "quote": quotes[qi],
            "start": start,
            "end": end,
            "fragment": text[start:end],
            "cost": cost,
            "sources": [list(src) for src in sources[start:end]],
        }

    # DP：处理完当前引文后，states[end] = [最低总代价, [以 end 结尾的链…]]，
    # 同一终点仅保留最低代价与字典序最小的前两条链。首段无衔接约束；之后要求
    # 前一引文区间终点 <= 当前区间起点（区间按序、互不重叠，可首尾相接）。
    states: dict[int, list] = {}
    for qi, candidates in enumerate(candidate_lists):
        nxt: dict[int, list] = {}
        for start, end, cost in candidates:
            if qi == 0:
                _merge_chains(nxt, end, cost, [[(start, end)]])
                continue
            prev = [(p_end, kept) for p_end, kept in states.items() if p_end <= start]
            if not prev:
                continue
            p_cost = min(kept[0] for _p, kept in prev)
            extended = [
                [*chain, (start, end)]
                for _p, kept in prev
                if kept[0] == p_cost
                for chain in kept[1]
            ]
            _merge_chains(nxt, end, p_cost + cost, extended)

        if not nxt:
            raise QuoteChainNotFoundError(
                f"第 {qi + 1} 段引文的全部合格区间都无法与前文引文的合格区间"
                "按序且互不重叠地衔接，不存在覆盖全部引文的完整定位链"
            )
        states = nxt

    best = min(kept[0] for kept in states.values())
    chains: list[list[tuple[int, int]]] = []
    for kept in states.values():
        if kept[0] == best:
            chains.extend(kept[1])
    chains.sort(key=_chain_key)
    unique: list[list[tuple[int, int]]] = []
    for chain in chains:
        if all(chain != other for other in unique):
            unique.append(chain)

    # 每段候选按区间索引的代价表，便于成链后回填各段命中的代价
    cost_by_interval = [
        {(s, e): c for s, e, c in candidates} for candidates in candidate_lists
    ]

    def make_chain(intervals: list[tuple[int, int]]) -> dict:
        hits = [
            make_hit(qi, s, e, cost_by_interval[qi][(s, e)])
            for qi, (s, e) in enumerate(intervals)
        ]
        return {"cost": best, "hits": hits}

    if len(unique) > 1:
        raise QuoteChainTieError(best, [make_chain(c) for c in unique[:2]])

    return {"text": text, "quotes": list(quotes), **make_chain(unique[0])}
