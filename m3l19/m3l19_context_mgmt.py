"""
第19课 核心模块：上下文生命周期管理

包含可单元测试的纯函数（bootstrap / prune / chunk / compress / session 持久化）
以及 XiaoPawCrew（Crew + Hooks）和 main（多轮演示入口）。

运行方式：
  修改文件底部 SESSION_ID / DEMO_ROUNDS 后直接执行
  cd m3l19 && python3 m3l19_context_mgmt.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

from crewai import Agent, Crew, LLM, Task
from crewai.hooks import LLMCallHookContext, before_llm_call
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, FileWriterTool, ScrapeWebsiteTool

# ── 项目根加入 sys.path，复用 llm / tools ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm import aliyun_llm                          # noqa: E402
from tools import BaiduSearchTool, FixedDirectoryReadTool  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 路径与常量
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = Path(__file__).parent / "workspace"
SESSIONS_DIR  = WORKSPACE_DIR / "sessions"

# 💡 可调整的核心参数
PRUNE_KEEP_TURNS   = 10      # 保留最近 N 轮原始 tool result，更早的清空占位
COMPRESS_THRESHOLD = 0.45    # 上下文使用率超过此值触发压缩
CHUNK_TOKENS       = 2000    # 压缩时每个 chunk 的近似 token 数
FRESH_KEEP_TURNS   = 10      # 压缩时保留最近 N 轮不压缩
MODEL_CTX_LIMIT    = 32000   # fallback：qwen3-max context window


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap：加载 workspace 文件构建 backstory
# ─────────────────────────────────────────────────────────────────────────────

def build_bootstrap_prompt(workspace_dir: Path) -> str:
    """
    💡 核心设计：只加载"导航骨架"，不把所有文件塞进去。
    soul（身份风格）+ user_profile（用户画像）
    + agent_rules（SOP 知识库）+ memory_index（记忆索引，200 行上限）
    """
    parts: list[str] = []

    for fname, tag in [
        ("soul.md",   "soul"),
        ("user.md",   "user_profile"),
        ("agent.md",  "agent_rules"),
    ]:
        path = workspace_dir / fname
        if path.exists():
            parts.append(
                f"<{tag}>\n{path.read_text(encoding='utf-8').strip()}\n</{tag}>"
            )

    # memory.md 限制 200 行，防膨胀
    memory_path = workspace_dir / "memory.md"
    if memory_path.exists():
        lines = memory_path.read_text(encoding="utf-8").splitlines()[:200]
        parts.append(
            f"<memory_index>\n{chr(10).join(lines)}\n</memory_index>"
        )

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Session 持久化（两份文件）
# ─────────────────────────────────────────────────────────────────────────────

def load_session_ctx(session_id: str, sessions_dir: Path = SESSIONS_DIR) -> list[dict]:
    """读取压缩 context 快照，用于 session 恢复"""
    p = sessions_dir / f"{session_id}_ctx.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_session_ctx(
    session_id: str,
    messages: list[dict],
    sessions_dir: Path = SESSIONS_DIR,
) -> None:
    """覆盖写入当前压缩 context 快照"""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}_ctx.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_session_raw(
    session_id: str,
    role: str,
    content: str,
    sessions_dir: Path = SESSIONS_DIR,
) -> None:
    """追加一条记录到原始完整历史（append-only，保留所有中间过程）"""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "role":    role,
        "content": content,
        "ts":      datetime.datetime.now().isoformat(),
    }
    with open(sessions_dir / f"{session_id}_raw.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 剪枝
# ─────────────────────────────────────────────────────────────────────────────

def prune_tool_results(
    messages: list[dict],
    keep_turns: int = PRUNE_KEEP_TURNS,
) -> None:
    """
    💡 核心设计：in-place 修改，Tool Result 是上下文膨胀的主要来源。
    策略：找到倒数第 keep_turns 个 user 消息的位置，
    该位置之前的所有 tool 消息内容替换为 [已剪枝]，保留消息占位和 tool_call_id。
    """
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_turns:
        return  # 轮数不足，无需剪枝

    cutoff_idx = user_indices[-keep_turns]   # 保留点：倒数第 N 个 user 消息
    for i in range(cutoff_idx):
        if messages[i].get("role") == "tool":
            messages[i]["content"] = "[已剪枝]"


# ─────────────────────────────────────────────────────────────────────────────
# 4. 压缩
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = """\
将以下对话历史压缩为结构化摘要，只保留关键信息：
1. 用户目标：这段对话要完成什么
2. 关键事实：重要的结论、文件路径、操作结果
3. 未完成事项：尚未完成的任务（如有）

禁止包含：中间过程、失败尝试、重复内容。

对话历史：
{history}
"""


def chunk_by_tokens(
    messages: list[dict],
    chunk_tokens: int = CHUNK_TOKENS,
) -> list[list[dict]]:
    """
    💡 按近似 token 数切分消息列表。
    估算方式：中文 1 字 ≈ 1 token，英文 4 字 ≈ 1 token，取保守值 len // 2。
    切分策略：当前 chunk 加入下一条消息后超过阈值时，先 flush 当前 chunk。
    单条消息超阈值时独立成 chunk（不截断消息内容）。
    """
    if not messages:
        return []

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = len(str(msg.get("content", ""))) // 2
        if current_tokens + msg_tokens > chunk_tokens and current:
            chunks.append(current)
            current = [msg]
            current_tokens = msg_tokens
        else:
            current.append(msg)
            current_tokens += msg_tokens

    if current:
        chunks.append(current)

    return chunks


def _summarize_chunk(messages: list[dict]) -> str:
    """💡 用轻量模型生成摘要，节省成本（可 mock 用于测试）"""
    summary_llm = LLM(model="qwen3-turbo")
    history = "\n".join(
        f"{m.get('role', '')}: {str(m.get('content', ''))[:300]}"
        for m in messages
    )
    return summary_llm.call([
        {"role": "user", "content": _SUMMARY_PROMPT.format(history=history)}
    ])


def maybe_compress(
    messages: list[dict],
    context: LLMCallHookContext,
) -> None:
    """
    💡 核心设计：in-place 修改 messages。
    超过 COMPRESS_THRESHOLD 时：
      ① 把旧消息按 CHUNK_TOKENS 分 chunk
      ② 每 chunk 单独调 qwen3-turbo 生成摘要
      ③ 用摘要（system 角色）替换原消息
    保留：原 system 消息 + 最近 FRESH_KEEP_TURNS 轮原文。
    """
    model_limit   = getattr(context.llm, "context_window_size", MODEL_CTX_LIMIT)
    approx_tokens = sum(len(str(m.get("content", ""))) // 2 for m in messages)
    if approx_tokens / model_limit < COMPRESS_THRESHOLD:
        return

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system  = [m for m in messages if m.get("role") != "system"]

    user_indices = [i for i, m in enumerate(non_system) if m.get("role") == "user"]
    if len(user_indices) <= FRESH_KEEP_TURNS:
        return  # 不足以压缩，跳过

    cutoff     = user_indices[-FRESH_KEEP_TURNS]
    old_msgs   = non_system[:cutoff]
    fresh_msgs = non_system[cutoff:]

    # 分 chunk，每 chunk 独立摘要
    chunks       = chunk_by_tokens(old_msgs, CHUNK_TOKENS)
    summary_msgs = [
        {
            "role":    "system",
            "content": f"<context_summary>\n{_summarize_chunk(chunk)}\n</context_summary>",
        }
        for chunk in chunks
    ]

    # in-place 替换：保留 system + 摘要 + 新鲜内容
    messages.clear()
    messages.extend(system_msgs + summary_msgs + fresh_msgs)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Crew（@CrewBase 提供 hook 绑定机制）
# ─────────────────────────────────────────────────────────────────────────────

@CrewBase
class XiaoPawCrew:
    """
    XiaoPaw 个人助手（第19课）
    演示：Bootstrap + @before_llm_call（剪枝 + 压缩）+ @after_llm_call（session 持久化）
    """

    def __init__(self, session_id: str, user_message: str) -> None:
        self.session_id      = session_id
        self.user_message    = user_message
        self._session_loaded = False  # 💡 session 恢复只做一次（首次 LLM 调用前）

    @agent
    def assistant_agent(self) -> Agent:
        return Agent(
            role      = "XiaoPaw 个人助手",
            goal      = "帮助晓寒高效完成各类任务，严谨、结果导向",
            backstory = build_bootstrap_prompt(WORKSPACE_DIR),  # 💡 Bootstrap 在这里
            llm       = aliyun_llm.AliyunLLM(
                model   = "qwen3-max",
                api_key = os.getenv("QWEN_API_KEY"),
                region  = "cn",
            ),
            tools = [
                BaiduSearchTool(),
                ScrapeWebsiteTool(),
                FileWriterTool(),
                FileReadTool(),
                FixedDirectoryReadTool(directory=str(WORKSPACE_DIR)),
            ],
            verbose  = True,
            max_iter = 50,
        )

    @task
    def assistant_task(self) -> Task:
        return Task(
            description     = "{user_request}",
            expected_output = "针对用户请求的完整回复",
            agent           = self.assistant_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents  = self.agents,
            tasks   = self.tasks,
            verbose = True,
        )

    # ── Pre-LLM Hook：session 恢复 + 剪枝 + 压缩 ─────────────────────────────

    @before_llm_call
    def before_llm_hook(self, context: LLMCallHookContext) -> bool | None:
        """
        💡 每次 LLM 调用前拦截，in-place 修改 context.messages。

        首次调用：从 ctx.json 恢复历史 context，追加本轮 user 消息。
        每次调用：① 剪枝旧 tool result → ② 超阈值时压缩。

        ⚠️ 为什么不用 @after_llm_call：
        crewai 的 _setup_after_llm_call_hooks 在 executor 有 after_llm_call_hooks 时，
        会把 answer（包括 tool_calls list）str() 化后返回，
        导致 executor 的 isinstance(answer, list) 判断失败，工具永远不执行。
        因此改为在下一次 before_llm_call 时，从 context.messages 里读取
        上一轮 executor 已追加的 assistant 消息来持久化。
        """
        if not self._session_loaded:
            self._restore_session(context)
            self._session_loaded = True
        else:
            # ── 非首次调用：把 executor 已追加的最新 assistant 消息写入 raw log ──
            # executor 在每次 LLM 返回后会 _append_message，所以 messages 末尾
            # 可能有新的 assistant / tool 消息，找到最后一条 assistant 消息记录
            last_assistant = next(
                (m for m in reversed(context.messages) if m.get("role") == "assistant"),
                None,
            )
            if last_assistant:
                content = last_assistant.get("content") or ""
                # 只记录文字回复（非 tool call 占位）
                if content and not str(content).strip().startswith("[{"):
                    append_session_raw(self.session_id, "assistant", str(content))
                    # 同步更新 ctx 快照（含本轮完整 messages）
                    save_session_ctx(self.session_id, list(context.messages))

        prune_tool_results(context.messages)     # ① 剪枝
        maybe_compress(context.messages, context)  # ② 压缩
        return None  # 返回 None 继续调用；返回 False 则阻止 LLM 调用

    # ── 内部：session 恢复 ────────────────────────────────────────────────────

    def _restore_session(self, context: LLMCallHookContext) -> None:
        """
        💡 取出 CrewAI 渲染的 user 消息 → 写入 raw log
        → 用历史 ctx 替换 context.messages + 追加新 user 消息。
        Agent 感知不到 session 中断，看到的是连续的上下文。
        """
        # 取出当前轮 user 消息（CrewAI 将 task description 渲染后注入，在末尾）
        current_user_msg = next(
            (m for m in reversed(context.messages) if m.get("role") == "user"),
            {},
        )
        if current_user_msg:
            append_session_raw(
                self.session_id,
                "user",
                str(current_user_msg.get("content", "")),
            )

        history = load_session_ctx(self.session_id)
        if not history:
            return  # 全新 session，无历史

        # 💡 替换：历史 context + 新 user 消息 = Agent 看到连续上下文
        context.messages.clear()
        context.messages.extend(history)
        if current_user_msg:
            context.messages.append(current_user_msg)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 运行参数 & Main（多轮演示）
# ─────────────────────────────────────────────────────────────────────────────

SESSION_ID = "demo"

DEMO_ROUNDS = [
    (
        "调研任务",
        "帮我调研极客时间平台上多智能体相关课程的现状，生成一份调研报告保存到文件",
    ),
    (
        "结论提炼",
        "把刚才报告里的关键结论总结成3条，方便我发给同事",
    ),
    (
        "周报生成",
        "帮我写本周工作总结，从记忆文件里读本周做了什么，保存到文件",
    ),
]


def main() -> None:
    ctx_file = SESSIONS_DIR / f"{SESSION_ID}_ctx.json"

    print(f"\n{'='*60}")
    print("XiaoPaw 助手 - 第19课：上下文生命周期管理")
    print(f"{'='*60}")
    print(f"Session ID : {SESSION_ID}")
    if ctx_file.exists():
        saved = json.loads(ctx_file.read_text())
        print(f"历史消息   : {len(saved)} 条（将恢复上下文）")
    else:
        print("历史消息   : 无（全新 session）")

    for i, (label, message) in enumerate(DEMO_ROUNDS, 1):
        print(f"\n{'─'*60}")
        print(f"Round {i}/{len(DEMO_ROUNDS)}  [{label}]")
        print(f"用户消息   : {message}")
        print(f"{'─'*60}\n")

        result = XiaoPawCrew(SESSION_ID, message).crew().kickoff(
            inputs={"user_request": message}
        )

        print(f"\n{'─'*60}")
        print(f"回复：\n{result.raw}")

    print(f"\n{'='*60}")
    print("Session 文件：")
    print(f"  ctx  → {SESSIONS_DIR / f'{SESSION_ID}_ctx.json'}")
    print(f"  raw  → {SESSIONS_DIR / f'{SESSION_ID}_raw.jsonl'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
