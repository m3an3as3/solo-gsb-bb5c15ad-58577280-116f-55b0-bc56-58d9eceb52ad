"""多段引文链式定位（locate_quotes）测试。"""

import itertools
import random

import pytest

from app import quote_chain
from app.assembler import AssemblyError
from app.locator import MAX_COST, _pair_cost
from app.quote_chain import (
    MAX_QUOTES,
    MIN_QUOTES,
    QuoteChainNotFoundError,
    QuoteChainTieError,
    locate_quotes,
)

SEGMENTS = ["学而?习之", "时习之不", "之不亦乐乎"]
# 拼接结果：学而时习之不亦乐乎（9 字）
TEXT = "学而时习之不亦乐乎"


def test_unique_chain_returns_each_interval():
    r = locate_quotes(SEGMENTS, ["时习之", "不亦乐乎"])
    assert r["text"] == TEXT
    assert r["quotes"] == ["时习之", "不亦乐乎"]
    assert r["cost"] == 0
    assert [h["quote"] for h in r["hits"]] == ["时习之", "不亦乐乎"]
    assert [(h["start"], h["end"]) for h in r["hits"]] == [(2, 5), (5, 9)]
    assert [h["fragment"] for h in r["hits"]] == ["时习之", "不亦乐乎"]
    assert [h["cost"] for h in r["hits"]] == [0, 0]
    # 各区间逐字 sources 与拼接结果对应区间一致
    assert r["hits"][0]["sources"] == [
        [{"segment": 0, "pos": 2}, {"segment": 1, "pos": 0}],
        [{"segment": 0, "pos": 3}, {"segment": 1, "pos": 1}],
        [{"segment": 0, "pos": 4}, {"segment": 1, "pos": 2}, {"segment": 2, "pos": 0}],
    ]
    assert len(r["hits"][1]["sources"]) == 4


def test_non_minimal_candidate_of_a_quote_must_join_chain():
    # 第 1 段引文 "甲" 的单独最优区间在 [4,5)（其后没有 "丙"，无法接入完整链），
    # 代价同为 0 的 [0,1) 才能与第 2 段引文的区间衔接，完整链必须选后者。
    r = locate_quotes(["甲X丙丁甲乙", "甲X丙丁甲乙"], ["甲", "丙"])
    assert r["text"] == "甲X丙丁甲乙"
    assert r["cost"] == 0
    assert [(h["start"], h["end"], h["fragment"]) for h in r["hits"]] == [
        (0, 1, "甲"),
        (2, 3, "丙"),
    ]


def test_chain_sums_per_quote_costs_and_picks_minimum():
    # 总代价 = 各段引文代价之和：第 1 段 0 + 第 2 段替换 1 = 1；
    # 第 2 段 "不亦X乎" 的唯一代价 1 区间为 [5,9)"不亦乐乎"
    r = locate_quotes(SEGMENTS, ["时习之", "不亦X乎"])
    assert r["cost"] == 1
    assert [(h["start"], h["end"], h["cost"]) for h in r["hits"]] == [
        (2, 5, 0),
        (5, 9, 1),
    ]


def test_intervals_may_be_adjacent_but_not_overlap():
    # 首尾相接（前一区间终点 == 后一区间起点）允许
    r = locate_quotes(SEGMENTS, ["时习之", "不亦乐乎"])
    assert [(h["start"], h["end"]) for h in r["hits"]] == [(2, 5), (5, 9)]
    # 逆序只有重叠区间可命中：无完整链
    with pytest.raises(QuoteChainNotFoundError):
        locate_quotes(SEGMENTS, ["不亦乐乎", "时习之"])


def test_tie_returns_two_lexicographically_smallest_chains():
    # 丁甲戊甲丙庚 中 "甲"→"丙" 恰有两条代价 0 链
    with pytest.raises(QuoteChainTieError) as exc:
        locate_quotes(["丁甲戊甲丙", "甲丙庚"], ["甲", "丙"])
    assert exc.value.cost == 0
    chains = [[(h["start"], h["end"]) for h in t["hits"]] for t in exc.value.ties]
    assert chains == [[(1, 2), (4, 5)], [(3, 4), (4, 5)]]
    for tie in exc.value.ties:
        assert tie["cost"] == 0
        assert [h["quote"] for h in tie["hits"]] == ["甲", "丙"]
        assert [h["cost"] for h in tie["hits"]] == [0, 0]
        # 两个 "甲" 分别在第 1 段下标 1、3（重叠位置可携带多段来源）
        assert {"segment": 0, "pos": 1} in tie["hits"][0]["sources"][0] or \
            {"segment": 0, "pos": 3} in tie["hits"][0]["sources"][0]


def test_tie_with_more_than_two_optimal_chains_lists_first_two():
    # 丁甲丙戊甲丙庚 中 "甲"→"丙" 有三条代价 0 链：
    # [(1,2),(2,3)] / [(1,2),(5,6)] / [(4,5),(5,6)]，只回前两条（字典序）
    with pytest.raises(QuoteChainTieError) as exc:
        locate_quotes(["丁甲丙戊甲丙", "甲丙庚"], ["甲", "丙"])
    chains = [[(h["start"], h["end"]) for h in t["hits"]] for t in exc.value.ties]
    assert chains == [[(1, 2), (2, 3)], [(1, 2), (5, 6)]]


def test_three_and_four_quotes_supported():
    r = locate_quotes(SEGMENTS, ["学", "习", "乐"])
    assert [(h["start"], h["end"]) for h in r["hits"]] == [(0, 1), (3, 4), (7, 8)]
    r = locate_quotes(SEGMENTS, ["学", "时", "习", "乎"])
    assert [(h["start"], h["end"]) for h in r["hits"]] == [
        (0, 1), (2, 3), (3, 4), (8, 9),
    ]


def test_quote_without_any_qualified_interval_raises():
    # 第 2 段引文与任何区间代价都远超 2
    with pytest.raises(QuoteChainNotFoundError):
        locate_quotes(SEGMENTS, ["时习之", "朋朋朋"])


def test_no_complete_chain_when_cannot_align_in_order():
    # 两段引文各自有代价 0 的区间，但全部 "戊己庚" 区间都在 "甲乙丙" 之前，
    # 不存在按序且互不重叠的完整链
    with pytest.raises(QuoteChainNotFoundError):
        locate_quotes(
            ["戊己庚丁甲乙丙", "戊己庚丁甲乙丙"], ["甲乙丙", "戊己庚"],
        )


def test_assembly_gate_failure_propagates():
    with pytest.raises(AssemblyError) as exc:
        locate_quotes(
            ["孔子曰未知", "未知生焉", "完全无关文本"], ["孔子", "未知"],
        )
    assert exc.value.failed_segment == 2


def test_quote_count_validated():
    with pytest.raises(ValueError):
        locate_quotes(SEGMENTS, ["时习之"])                    # 仅 1 段
    with pytest.raises(ValueError):
        locate_quotes(SEGMENTS, ["学", "习", "亦", "乐", "乎"])  # 5 段
    assert MIN_QUOTES == 2 and MAX_QUOTES == 4


def test_quote_length_validated():
    with pytest.raises(ValueError):
        locate_quotes(SEGMENTS, ["", "时习之"])
    with pytest.raises(ValueError):
        locate_quotes(SEGMENTS, ["时习之", "学" * 81])
    # 恰好 80 字可通过参数与拼接门槛（具体命中/并列/无链取决于文本）
    try:
        locate_quotes(["学" * 200 + "时习之", "学" * 200 + "时习之"], ["?" * 80, "时习之"])
    except (QuoteChainNotFoundError, QuoteChainTieError):
        pass


# ---------------------------------------------------------------------------
# 暴力全枚举校验：随机文本与引文，穷举全部区间组合，对比最优链、并列与无链判定
# ---------------------------------------------------------------------------

def _brute_candidates(text, quote):
    out = []
    m = len(quote)
    for s in range(len(text)):
        for e in range(s + 1, len(text) + 1):
            frag = text[s:e]
            n = len(frag)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                ndp = [i] + [0] * n
                for j in range(1, n + 1):
                    pc = _pair_cost(quote[i - 1], frag[j - 1])
                    ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + pc)
                dp = ndp
            if dp[n] <= MAX_COST:
                out.append((s, e, dp[n]))
    return out


def test_chain_dp_matches_brute_force(monkeypatch):
    random.seed(7)
    pool = "甲乙丙?"
    for _ in range(300):
        text = "".join(random.choice(pool) for _ in range(random.randint(2, 10)))
        k = random.randint(2, 3)
        quotes = [
            "".join(random.choice(pool) for _ in range(random.randint(1, 4)))
            for _ in range(k)
        ]
        cands = [_brute_candidates(text, q) for q in quotes]
        best, best_chains = None, []
        if all(cands):
            for combo in itertools.product(*cands):
                if all(combo[i][1] <= combo[i + 1][0] for i in range(k - 1)):
                    total = sum(c[2] for c in combo)
                    iv = [(c[0], c[1]) for c in combo]
                    if best is None or total < best:
                        best, best_chains = total, [iv]
                    elif total == best and iv not in best_chains:
                        best_chains.append(iv)
        best_chains.sort()

        fake_sources = [[{"segment": 0, "pos": i}] for i in range(len(text))]
        monkeypatch.setattr(
            quote_chain, "assemble",
            lambda _segments: {"text": text, "sources": fake_sources},
        )
        if best is None:
            with pytest.raises(QuoteChainNotFoundError):
                locate_quotes(["__a__", "__b__"], quotes)
            continue
        if len(best_chains) == 1:
            r = locate_quotes(["__a__", "__b__"], quotes)
            assert r["cost"] == best
            assert [(h["start"], h["end"]) for h in r["hits"]] == best_chains[0]
        else:
            with pytest.raises(QuoteChainTieError) as exc:
                locate_quotes(["__a__", "__b__"], quotes)
            assert exc.value.cost == best
            chains = [
                [(h["start"], h["end"]) for h in t["hits"]]
                for t in exc.value.ties
            ]
            assert chains == best_chains[:2]
