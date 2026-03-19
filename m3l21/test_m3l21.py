"""
test_m3l21.py — 第21课单测

测试范围：
  1. parse_turns：session JSONL 解析
  2. extract_summary_and_tags：LLM 提取（mock）
  3. embed_texts：向量化（mock）
  4. upsert_memory：数据库写入（mock）
  5. search：三种搜索模式（mock DB）
  6. async_index_turn：异步建索引（mock）
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── 把 m3l21 目录加入 sys.path ────────────────────────────────────────────────
import sys
_M3L21 = Path(__file__).resolve().parent
if str(_M3L21) not in sys.path:
    sys.path.insert(0, str(_M3L21))


# ─────────────────────────────────────────────────────────────────────────────
# 1. parse_turns
# ─────────────────────────────────────────────────────────────────────────────

def _make_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "test_session.jsonl"
    p.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8")
    return p


def test_parse_turns_basic(tmp_path):
    """正常两轮对话，应解析出 2 个 turn"""
    from indexer import parse_turns

    lines = [
        {"type": "meta", "session_id": "s001", "routing_key": "p2p:ou_abc"},
        {"type": "message", "role": "user",      "content": "帮我查航班", "ts": 1000},
        {"type": "message", "role": "assistant", "content": "已查到航班信息", "ts": 1001},
        {"type": "message", "role": "user",      "content": "帮我转PDF", "ts": 2000},
        {"type": "message", "role": "assistant", "content": "转换完成", "ts": 2001},
    ]
    turns = parse_turns(_make_jsonl(tmp_path, lines))

    assert len(turns) == 2
    assert turns[0]["user_message"]    == "帮我查航班"
    assert turns[0]["assistant_reply"] == "已查到航班信息"
    assert turns[0]["session_id"]      == "s001"
    assert turns[0]["routing_key"]     == "p2p:ou_abc"
    assert turns[1]["user_message"]    == "帮我转PDF"


def test_parse_turns_no_meta(tmp_path):
    """没有 meta 行时，routing_key 应降级为 unknown"""
    from indexer import parse_turns

    lines = [
        {"type": "message", "role": "user",      "content": "你好", "ts": 1000},
        {"type": "message", "role": "assistant", "content": "你好！", "ts": 1001},
    ]
    turns = parse_turns(_make_jsonl(tmp_path, lines))
    assert len(turns) == 1
    assert turns[0]["routing_key"] == "unknown"


def test_parse_turns_trailing_user(tmp_path):
    """最后一条 user 消息没有对应 assistant，应忽略"""
    from indexer import parse_turns

    lines = [
        {"type": "meta", "session_id": "s002", "routing_key": "p2p:ou_abc"},
        {"type": "message", "role": "user",      "content": "第一问", "ts": 1000},
        {"type": "message", "role": "assistant", "content": "第一答", "ts": 1001},
        {"type": "message", "role": "user",      "content": "未回答的问题", "ts": 2000},
    ]
    turns = parse_turns(_make_jsonl(tmp_path, lines))
    assert len(turns) == 1


def test_parse_turns_empty(tmp_path):
    """空文件应返回空列表"""
    from indexer import parse_turns

    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    turns = parse_turns(p)
    assert turns == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. extract_summary_and_tags（mock LLM）
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_summary_and_tags_normal():
    """正常 JSON 返回，应正确解析摘要和标签"""
    from indexer import extract_summary_and_tags

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps(
        {"summary": "帮用户查了航班信息", "tags": ["工作", "出行"]},
        ensure_ascii=False,
    )

    with patch("indexer._llm_client.chat.completions.create", return_value=mock_resp):
        summary, tags = extract_summary_and_tags("帮我查航班", "已查到航班信息")

    assert summary == "帮用户查了航班信息"
    assert tags    == ["工作", "出行"]


def test_extract_summary_and_tags_markdown_wrapped():
    """LLM 返回 markdown 代码块包裹的 JSON，应正确解析"""
    from indexer import extract_summary_and_tags

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "```json\n{\"summary\": \"PDF转换\", \"tags\": [\"文件处理\"]}\n```"

    with patch("indexer._llm_client.chat.completions.create", return_value=mock_resp):
        summary, tags = extract_summary_and_tags("帮我转PDF", "转换完成")

    assert summary == "PDF转换"
    assert tags    == ["文件处理"]


def test_extract_summary_and_tags_invalid_json():
    """LLM 返回非法 JSON 时，应降级返回 user_message 前50字符和空标签"""
    from indexer import extract_summary_and_tags

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "这不是JSON"

    with patch("indexer._llm_client.chat.completions.create", return_value=mock_resp):
        summary, tags = extract_summary_and_tags("帮我查航班信息", "已查到")

    assert "帮我查航班信息" in summary
    assert tags == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. embed_texts（mock embedding API）
# ─────────────────────────────────────────────────────────────────────────────

def test_embed_texts_returns_correct_shape():
    """embed_texts 应返回与输入等长的向量列表，每个向量维度为 EMBED_DIM"""
    from indexer import embed_texts, EMBED_DIM

    fake_vec = [0.1] * EMBED_DIM
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=fake_vec), MagicMock(embedding=fake_vec)]

    with patch("indexer._embed_client.embeddings.create", return_value=mock_resp):
        vecs = embed_texts(["文本一", "文本二"])

    assert len(vecs)    == 2
    assert len(vecs[0]) == EMBED_DIM


# ─────────────────────────────────────────────────────────────────────────────
# 4. upsert_memory（mock DB）
# ─────────────────────────────────────────────────────────────────────────────

def test_upsert_memory_calls_execute():
    """upsert_memory 应调用 cursor.execute 并 commit"""
    from indexer import upsert_memory, EMBED_DIM

    mock_conn   = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)

    record = {
        "id":              "test_id",
        "session_id":      "s001",
        "routing_key":     "p2p:ou_abc",
        "user_message":    "帮我查航班",
        "assistant_reply": "已查到",
        "summary":         "查航班",
        "tags":            ["出行"],
        "turn_ts":         1000,
        "summary_vec":     [0.1] * EMBED_DIM,
        "message_vec":     [0.2] * EMBED_DIM,
        "search_text":     "帮我查航班 出行",
    }

    upsert_memory(mock_conn, record)

    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 5. search 三种模式（mock DB + mock embed）
# ─────────────────────────────────────────────────────────────────────────────

def _mock_search_setup(mode: str):
    """公共 mock 设置，返回一条假记录"""
    import datetime

    fake_vec  = [0.1] * 1024
    fake_row  = {
        "id":              "abc",
        "summary":         "查航班",
        "user_message":    "帮我查航班",
        "assistant_reply": "已查到",
        "tags":            ["出行"],
        "created_at":      datetime.datetime(2026, 1, 20, 14, 0, 0),
        "turn_ts":         1000,
        "score":           0.92,
    }

    mock_embed_resp = MagicMock()
    mock_embed_resp.data = [MagicMock(embedding=fake_vec)]

    mock_conn   = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [fake_row]
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)

    return mock_conn, mock_embed_resp, fake_row


@pytest.mark.parametrize("mode", ["hybrid", "vector", "fulltext"])
def test_search_modes(mode):
    """三种搜索模式都应返回结果列表，且 score 字段存在"""
    import sys
    # search.py 在 skills/search_memory/scripts/ 下，需要加入 path
    scripts_dir = str(_M3L21 / "skills" / "search_memory" / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import importlib
    search_mod = importlib.import_module("search")

    mock_conn, mock_embed_resp, _ = _mock_search_setup(mode)

    with patch("search._embed_client.embeddings.create", return_value=mock_embed_resp), \
         patch("search.psycopg2.connect", return_value=mock_conn):
        results = search_mod.search(query="航班", mode=mode, limit=5)

    assert isinstance(results, list)
    assert len(results) == 1
    assert "score" in results[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. async_index_turn（mock 整个 _index_single_turn）
# ─────────────────────────────────────────────────────────────────────────────

def test_async_index_turn_calls_single_turn():
    """async_index_turn 应在 executor 中调用 _index_single_turn"""
    from indexer import async_index_turn

    with patch("indexer._index_single_turn") as mock_fn:
        asyncio.run(async_index_turn(
            session_id      = "s001",
            routing_key     = "p2p:ou_abc",
            user_message    = "帮我查航班",
            assistant_reply = "已查到",
            turn_ts         = 1000,
        ))
        mock_fn.assert_called_once_with(
            "s001", "p2p:ou_abc", "帮我查航班", "已查到", 1000
        )
