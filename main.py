"""
astrbot_plugin_proactive_reply —— AstrBot 智能自主回复插件

功能特性：
1. 私聊 / 群聊「自主回复」独立开关；群聊支持识别非 @ 消息，并按冷场概率/关键词自主决定是否回复；
2. 上下文延续判断：让 LLM 自动识别消息是否在与机器人对话，无关的消息不回复（仅群聊非@自主回复生效）；
3. 自动联系上下文回复（按会话维度独立维护记忆，仅记录机器人参与过的对话）；
4. 关键词触发回复（必回）+ 冷场概率（只影响无进行中对话时的主动搭话，不影响持续对话）；
5. 免打扰时间段（格式 "22:00-9:00"，支持跨天）；免打扰时段内可分别控制「自主回复」和「@回复」是否放行；
6. 私聊主动问候：私聊长时间未聊天（默认 3 天，可自定义）时主动发消息，内容按模型人设生成；
7. 群聊回复末尾附带被回复人昵称（可开关）；
8. 插件运行日志收集（滚动文件日志 + AstrBot 控制台日志），便于后期排查问题。

说明：v1.7.0 起，本插件在自行调用 LLM 前后会手动触发 AstrBot 的
on_llm_request / on_llm_response 钩子链，因此 lili_state 等通过
@on_llm_request 注入人设/状态的插件现在能正常作用于本插件发出的请求。
钩子不可用时自动降级为旧行为（不触发钩子），不影响基本功能。
依赖：无第三方依赖，仅使用 Python 标准库与 AstrBot 官方 API。
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections import deque
from datetime import datetime, time as dtime
from difflib import SequenceMatcher
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from astrbot.api import AstrBotConfig
from astrbot.api.all import MessageChain
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star, register

# AstrBot 提供的日志接口，会根据调用位置自动路由到当前插件的 logger。
# 如果导入失败（理论不会发生），回退到标准库 logger。
try:
    from astrbot.api import logger as astr_logger
except Exception:  # pragma: no cover
    astr_logger = logging.getLogger("astrbot_plugin_proactive_reply")

# ---------------------------------------------------------------------------
# 兼容导入：手动触发 on_llm_request / on_llm_response 钩子链。
# 这样 lili_state 等「人设/状态管理」插件就能像在主 pipeline 里一样，
# 在我们自己调 LLM 之前注入 system_prompt，并在拿到回复后更新状态。
# 导入失败（旧版本 AstrBot 或内部路径变动）时静默降级为原行为：不触发钩子。
# ---------------------------------------------------------------------------
try:
    from astrbot.core.pipeline.context_utils import call_event_hook
    from astrbot.core.star.star_handler import EventType
    from astrbot.core.provider.entities import ProviderRequest, LLMResponse
    _HOOKS_AVAILABLE = True
except Exception:  # pragma: no cover
    call_event_hook = None
    EventType = None
    ProviderRequest = None
    LLMResponse = None
    _HOOKS_AVAILABLE = False

PLUGIN_DIR = Path(__file__).resolve().parent
LOG_DIR = PLUGIN_DIR / "logs"
LOG_FILE = LOG_DIR / "auto_reply.log"

# 默认的上下文延续判断提示词：追加在当前用户消息之后，让模型判断是否继续对话。
# 模型若判定消息与机器人无关（或重复同一话题），应只输出 __NO_REPLY__，
# 插件将丢弃该回复且不写入记忆。
# 可在配置面板「上下文判断提示词」中自定义修改；此处为默认值。
# 措辞刻意「倾向延续」：只有明显与机器人无关的群友闲聊或重复同一件事才允许拒绝，
# 避免把正在继续的对话误判为无关（漏回），同时避免对意思差不多的消息重复回复。
# 特别要求模型「不要输出判断过程/确认文字」，防止判定结果被当作回复发出去。
DEFAULT_CONTEXT_JUDGE_INSTRUCTION = (
    "\n\n[群聊插话判断] 你是这个群的群友之一，现在判断上面这条「当前消息」你要不要插嘴说话。"
    "规则：\n"
    "【必须直接接话】（直接说出你要说的话）：\n"
    "1) 消息点名你、@你、直接问你；\n"
    "2) 这条消息是在回应你之前说过的话（接你的梗、答你的问题、跟你抬杠）；\n"
    "3) 发消息的人是你刚刚才回复过的，他在接着跟你聊。\n"
    "【可以选择性加入】（只有你确实有话想说、有观点想发表时才开口）：\n"
    "4) 群里在聊一个开放的、大家都能参与的话题（比如某部作品、某款游戏、某个热点），"
    "而且你作为群友也有独特看法或好玩的梗可以抛出来——这种情况下用你自己的人设语气自然加入。\n"
    "【一律不要插嘴】（只输出 __NO_REPLY__ 五个字符）：\n"
    "a) 这是两个人之间的一问一答、短对话回合（比如 A 问「在干嘛」B 答「在玩游戏」，"
    "这种一来一回的私聊式对话你不要插嘴）；\n"
    "b) 消息明显是回复某一个特定的人，跟你没关系；\n"
    "c) 话题跟你八竿子打不着，你也没什么自然可说的；\n"
    "d) 你不确定该不该说话时——默认不回。\n"
    "重要：你的输出就是直接说给群友听的话，不要输出任何判断过程、理由、「我应该/不应该」之类的元叙述。"
)

# 常见的「判定确认文本」模式：模型没有给出真实回复、而是回答判断结论
# （如“是，这条消息是在与我继续对话。”“是在与你继续对话。因为这条消息
# 点名了…属于明确与你互动的新问题。”）。识别到则丢弃，不发送。
_JUDGE_TEXT_PATTERNS = (
    # 模型只输出判断结论，如“是”“是的”“不是”
    re.compile(r"^(是|是的|不是|对|对的|不对|没错|没有|并没有)[，,。.！!？?、\s]*$"),
    # 模型复述判断：如“是，这条消息是在与我继续对话。”“这条消息是在与你对话”
    re.compile(r"^[是是不]?\s*[，,。.]?\s*(这|该|此)条消息.{0,60}(对话|聊天|说话|回复)"),
    # 模型判定无关：如“这条消息与我无关”“该消息不是在与您对话”
    re.compile(r"^(这|该|此)条消息.{0,50}(无关|不是|没有)(在)?(与|和)?.{0,20}(对话|聊天|说话)"),
    # 模型引用判断规则作答
    re.compile(r"^(根据|按照).{0,20}(判断|规则).{0,30}(不回复|__NO_REPLY__|无需|无关)"),
    # 实测输出格式：“是在与你继续对话。因为这条消息点名了…属于明确与你互动的新问题。”
    re.compile(r"^是在与你?(正在|继续|在)?[，,。.、\s]*对话"),
    re.compile(r"^(这|该|此)条消息.{0,30}(点名|提到|提出|延续|属于|判断|互动).{0,60}(新问题|新任务|新信息|互动|话题|对话|判断)"),
    re.compile(r"(因为|由于)?(这|该|此)条消息.{0,30}(点名|提到|提出).{0,40}(互动|问题|任务|话题|判断|新消息)"),
    re.compile(r"属于(明确)?(与你)?(互动|继续对话|判断).{0,30}(新问题|新任务|新信息|新消息)"),
    # 极短判断句式：“是在与你继续对话”“不是在与你对话”
    re.compile(r"^(是|不是|是在|不是在)?与你?.{0,6}(继续)?(对话|聊天)[。.!！]*$"),
)

# 防重复消息的时间窗口（秒）：同一会话内完全相同的文本在此窗口内再次出现则跳过
DUPLICATE_WINDOW_SEC = 60.0

# 私聊主动问候循环的检查间隔（秒）
GREETING_CHECK_INTERVAL_SEC = 1800.0

# 私聊主动问候的默认生成指令：内容由模型按其人设（其他插件/提供商配置）生成，
# 本插件不注入人设。
DEFAULT_GREETING_PROMPT = (
    "（主动问候）你注意到已经有一段时间没有和对方聊天了，现在想主动关心一下对方。"
    "请你按照你的人设和说话风格，给这位久未联系的朋友发一条自然、真诚的关心消息"
    "（例如关心对方最近过得怎么样、提醒注意休息和身体等）。"
    "要求：语气符合你的人设，长度一两句话，直接输出要发送的内容，"
    "不要输出任何解释、标题或多余文字。"
)

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


@register(
    "astrbot_plugin_proactive_reply",
    "Doubao",
    "智能自主回复插件：上下文延续判断、关键词触发、免打扰时段、独立开关与运行日志（v1.7.5 不插嘴别人的@对话）",
    "1.7.5",
    "https://github.com/AstrBotDevs/AstrBot",
)
class ProactiveReplyPlugin(Star):
    """智能自主回复插件主类。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 会话记忆：key = unified_msg_origin（会话唯一 ID）
        self._histories: dict[str, deque] = {}
        # 会话级异步锁，避免同一会话并发时记忆读写交错
        self._locks: dict[str, asyncio.Lock] = {}
        # 近期已处理消息（防重复回复）：key = umo，deque[(text, timestamp)]
        self._recent_msgs: dict[str, deque] = {}
        # 私聊会话最近活跃时间：key = umo，value = 最近一次收到消息的时间戳
        self._private_active: dict[str, float] = {}
        # 私聊主动问候发送记录：key = umo，value = 最近一次主动问候的时间戳
        self._greeting_sent: dict[str, float] = {}
        # 主动问候后台任务
        self._greeting_task: Optional[asyncio.Task] = None

        self._setup_file_logger()

        # 初始化后立即写一条启动日志，方便确认插件已加载
        self._log("INFO", f"插件加载完成，日志文件：{LOG_FILE}")

        # 启动私聊主动问候后台任务
        self._start_greeting_loop()

    # ------------------------------------------------------------------ #
    # 日志
    # ------------------------------------------------------------------ #
    def _setup_file_logger(self) -> None:
        """初始化插件文件日志（UTF-8，单文件最大 2MB，共保留最近 3 份）。"""
        self._file_logger: Optional[logging.Logger] = None
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            level_name = str(self.config.get("log_level", "INFO") or "INFO").upper()
            level = getattr(logging, level_name, logging.INFO)

            handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=2 * 1024 * 1024,
                backupCount=2,  # 当前文件 + 2 份轮换 = 只保留最近 3 份
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))

            self._file_logger = logging.getLogger("astrbot_plugin_proactive_reply.file")
            self._file_logger.setLevel(level)
            # 插件重载时清理旧 handler，避免重复写日志
            self._file_logger.handlers.clear()
            self._file_logger.addHandler(handler)
            self._file_logger.propagate = False
        except Exception as e:  # 日志失败不能影响插件主体功能
            astr_logger.error(f"[智能自主回复] 初始化文件日志失败: {e}")
            self._file_logger = None

    def _log(self, level: str, msg: str) -> None:
        """写一条日志：同时写入 AstrBot 日志（logs/astrbot.log，方便统一查看）
        与插件文件日志（保留最近 3 份）。"""
        text = f"[智能自主回复] {msg}"
        # 1) AstrBot 日志：优先使用 AstrBot 分配给本插件的 logger（会进入
        #    AstrBot 控制台与其日志文件），失败时回退到全局接口。
        try:
            logger = getattr(self, "logger", None) or astr_logger
            getattr(logger, level.lower(), logger.info)(text)
        except Exception:
            pass
        # 2) 插件文件日志（双保险，便于单独导出）
        try:
            if self._file_logger is not None:
                getattr(self._file_logger, level.lower(), self._file_logger.info)(msg)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 事件处理
    # ------------------------------------------------------------------ #
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent):
        """私聊消息处理器。"""
        reply = await self._handle(event, is_group=False)
        if reply:
            event.stop_event()  # 先标记停止，阻止后续（内置 AI 等）重复回复
            yield event.plain_result(reply)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """群聊消息处理器（同时覆盖 @ 消息与非 @ 消息）。"""
        reply = await self._handle(event, is_group=True)
        if reply:
            event.stop_event()  # 先标记停止，阻止后续（内置 AI 等）重复回复
            yield event.plain_result(reply)

    async def _handle(self, event: AstrMessageEvent, is_group: bool) -> Optional[str]:
        """自主回复核心逻辑：判断 -> 决策 -> 上下文延续判断 -> 调 LLM -> 返回回复文本。"""
        try:
            text = (event.message_str or "").strip()
            # 部分平台（如 QQ 官方接口）会把 @ 前缀带进文本，先清理
            text = re.sub(r"^@[\w\u4e00-\u9fa5_-]+\s*", "", text).strip()
            sender_name = self._safe_get(event, "get_sender_name", "未知用户")
            sender_id = self._safe_get(event, "get_sender_id", "")
            umo = event.unified_msg_origin

            self._log(
                "INFO",
                f"收到消息 | 类型={'群聊' if is_group else '私聊'} | 发送者={sender_name}({sender_id}) | 内容={text[:80] or '(无文本)'}",
            )

            # 没有可回复的文本内容（例如纯图片/语音）时直接跳过，交还给 AstrBot 默认流程
            if not text:
                self._log("INFO", "消息无文本内容，跳过自主回复")
                return None

            # 防重复：同一会话内，窗口期已处理过完全相同的文本则跳过
            # （可避免平台重复推送同一条消息 / 用户连发相同内容导致的重复回复）
            if self._is_recent_duplicate(umo, text):
                self._log(
                    "INFO",
                    f"检测到重复消息（{DUPLICATE_WINDOW_SEC:.0f}秒内相同内容已处理），跳过回复 | 内容={text[:50]}",
                )
                return None
            self._remember_msg(umo, text)

            is_at_me = self._is_at_me(event)
            has_at_other = is_group and (not is_at_me) and self._has_at_other(event)
            if is_group and is_at_me:
                self._log("INFO", "群聊消息命中 @ 机器人")
            if has_at_other:
                self._log("INFO", "群聊消息 @ 了别人，bot 不插嘴，跳过自主回复")
                return None

            # 私聊：记录会话最近活跃时间（用于「长时间未聊 → 主动问候」）
            if not is_group:
                self._private_active[umo] = time.time()

            # 对话延续信号：bot 刚回复过这位发送者 → 视为对话仍在继续，必回、不判断
            pending = self._has_pending_context(umo, sender_name)
            # 关键词触发信号：命中关键词 → 必回、不判断（触发即答）
            hit_keyword = self._hit_keyword(text)

            should, reason, block = self._should_reply(
                is_group, is_at_me, text, pending=pending, hit_keyword=hit_keyword
            )
            if not should:
                self._log("INFO", f"不回复 | 原因={reason}")
                if block:
                    # 免打扰拦截：阻止后续（内置 AI 等）继续回复
                    self._log("INFO", "免打扰拦截：阻止事件继续传播")
                    event.stop_event()
                return None

            self._log("INFO", f"决定回复 | 原因={reason}")

            # 会话记忆（加锁，防止同一会话并发写乱）
            lock = self._get_lock(umo)
            async with lock:
                history = self._get_history(umo)
                # 群聊中带上发送者昵称，帮助模型区分谁在说话
                display_text = f"{sender_name}: {text}" if is_group else text

                msgs = self._build_messages(history, display_text)
                # 上下文延续判断：仅对「群聊非 @ 自主回复、且非延续、且非关键词触发」生效
                need_judge = (
                    is_group
                    and not is_at_me
                    and self.config.get("context_aware_reply", True)
                    and not pending
                    and not hit_keyword
                )
                if pending:
                    self._log("INFO", "检测到延续信号（刚回复过该用户），直接继续回复")
                if hit_keyword and not is_at_me:
                    self._log("INFO", "命中关键词，按触发规则直接回复")
                reply, status = await self._llm_chat(event, umo, msgs, need_context_judge=need_judge)

                if status == "no_reply":
                    self._log("INFO", "上下文延续判断：该消息与机器人对话无关，不回复")
                    return None
                if not reply:
                    self._log("WARNING", "LLM 未返回有效回复，放弃本条消息")
                    return None

                # 群聊回复末尾附带被回复人昵称（可配置开关）
                if (
                    is_group
                    and self.config.get("append_nickname", True)
                    and sender_name
                    and sender_name != "未知用户"
                ):
                    reply = f"{reply}\n@{sender_name}"

                # 写入记忆（用户消息 + 机器人回复）
                history.append({"role": "user", "content": display_text})
                history.append({"role": "assistant", "content": reply})

            self._log("INFO", f"回复内容 | {reply[:100]}")
            return reply
        except Exception as e:
            self._log("ERROR", f"处理消息异常: {e!r}")
            return None

    # ------------------------------------------------------------------ #
    # 私聊主动问候（长时间未聊天 → 按人设主动发消息）
    # ------------------------------------------------------------------ #
    def _start_greeting_loop(self) -> None:
        """启动私聊主动问候后台任务（插件加载时调用，terminate 时取消）。"""
        try:
            if self._greeting_task is None:
                self._greeting_task = asyncio.create_task(self._greeting_loop())
                self._log("INFO", f"私聊主动问候后台任务已启动（检查间隔 {GREETING_CHECK_INTERVAL_SEC:.0f}s）")
        except Exception as e:
            self._log("WARNING", f"启动私聊主动问候任务失败: {e!r}")

    async def _greeting_loop(self) -> None:
        """后台循环：周期性检查是否有私聊会话需要主动问候。"""
        while True:
            try:
                await self._maybe_greet()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._log("ERROR", f"主动问候循环异常: {e!r}")
            try:
                await asyncio.sleep(GREETING_CHECK_INTERVAL_SEC)
            except asyncio.CancelledError:
                raise

    async def _maybe_greet(self) -> None:
        """检查各私聊会话，超过设定天数未聊天则按人设主动发送一条问候。"""
        if not self.config.get("private_greeting_enable", True):
            return
        try:
            days = max(1, int(self.config.get("private_greeting_days", 3) or 3))
        except Exception:
            days = 3
        threshold = days * 86400.0
        now = time.time()

        for umo, last_ts in list(self._private_active.items()):
            if now - last_ts < threshold:
                continue  # 还没到主动问候的时间
            last_greet = self._greeting_sent.get(umo, 0.0)
            if now - last_greet < threshold:
                continue  # 上次问候后还没再次超期，避免连续打扰

            self._log("INFO", f"检测到私聊会话长时间未互动（>{days}天），准备主动问候 | 会话={umo}")
            greeting = await self._generate_greeting(umo)
            if not greeting:
                self._log("WARNING", f"主动问候生成失败，本次跳过 | 会话={umo}")
                continue

            try:
                ok = await self.context.send_message(
                    umo, MessageChain([Plain(greeting)])
                )
                self._log(
                    "INFO",
                    f"主动问候已发送 | 会话={umo} | 发送成功={ok} | 内容={greeting[:80]}",
                )
                # 无论平台返回与否，都记录发送时间并重置活跃计数，避免重复打扰
                self._greeting_sent[umo] = now
                self._private_active[umo] = now
                await asyncio.sleep(1)  # 同一轮多个会话时避免发送过密
            except Exception as e:
                self._log("WARNING", f"主动问候发送失败 | 会话={umo} | {e!r}")

    async def _generate_greeting(self, umo: str) -> Optional[str]:
        """按「人设」生成主动问候内容：只传问候指令，不注入人设，
        由当前会话绑定的模型提供商的人设（其他插件配置）决定说话风格。"""
        prov = None
        provider_id = str(self.config.get("provider_id", "") or "").strip()
        if provider_id:
            try:
                getter = getattr(self.context, "get_provider_by_id", None)
                if callable(getter):
                    prov = getter(provider_id)
            except Exception:
                pass
        if prov is None and self.config.get("use_session_provider", True):
            getter_async = getattr(self.context, "get_using_provider_async", None)
            if callable(getter_async):
                try:
                    prov = await getter_async(umo=umo)
                except Exception:
                    pass
            if prov is None:
                getter_sync = getattr(self.context, "get_using_provider", None)
                if callable(getter_sync):
                    try:
                        prov = getter_sync()
                    except Exception:
                        pass
        if prov is None:
            self._log("ERROR", "主动问候：未找到可用的 LLM 提供商")
            return None

        prompt = str(self.config.get("private_greeting_prompt", "") or "").strip()
        if not prompt:
            prompt = DEFAULT_GREETING_PROMPT

        try:
            resp = await asyncio.wait_for(prov.text_chat(prompt=prompt), timeout=60)
            text = self._extract_completion(resp)
            if not text:
                return None
            text = text.strip()
            if self._looks_like_judge_output(text):
                return None
            return text
        except asyncio.TimeoutError:
            self._log("ERROR", "主动问候：LLM 调用超时（60s）")
        except Exception as e:
            self._log("WARNING", f"主动问候：LLM 调用失败: {e!r}")
        return None

    # ------------------------------------------------------------------ #
    # 决策逻辑
    # ------------------------------------------------------------------ #
    def _should_reply(
        self,
        is_group: bool,
        is_at_me: bool,
        text: str,
        pending: bool = False,
        hit_keyword: bool = False,
    ) -> tuple[bool, str, bool]:
        """根据开关、免打扰时段、关键词、概率等决定是否回复。

        返回 (是否回复, 原因, 是否拦截后续流程)。
        拦截仅发生在「免打扰时段判定不应回复」时：此时插件不回复，
        也要阻止 AstrBot 内置回复，让免打扰彻底生效。

        概率语义（v1.6.0）：回复概率只影响「冷场/新话题」场景——
        没有进行中的对话时，是否主动参与新话题（抽中则回，抽不中则不回）。
        正在进行的对话（延续信号 pending）与关键词触发、@ 消息均必回，
        不受概率影响，保证“聊着聊着就不接话”不再发生。
        """
        # 1. 插件总开关
        if not self.config.get("enable_plugin", True):
            return False, "插件总开关关闭", False

        # 2. 是否处于免打扰时段
        is_quiet = self.config.get("quiet_periods_enabled", True) and self._in_quiet_period(
            datetime.now().time()
        )

        # 3. 群聊
        if is_group:
            if is_at_me:
                # @ 消息：无独立开关，非免打扰时段总是回复；
                # 免打扰时段内由 quiet_at_reply 控制是否放行。
                if is_quiet and not self.config.get("quiet_at_reply", False):
                    return False, "免打扰时段@回复开关关闭", True
                return True, "群聊@消息", False

            # 非 @ 消息：群聊自主回复
            if not self.config.get("group_chat_auto_reply", False):
                return False, "群聊自主回复开关关闭（非@消息不回复）", False
            if is_quiet and not self.config.get("quiet_auto_reply", False):
                return False, "免打扰时段自主回复开关关闭", True

            # 关键词触发：必回
            if hit_keyword:
                return True, "命中关键词", False
            # 进行中的对话（刚回复过该用户）：必回，不受概率影响
            if pending:
                return True, "对话延续中（刚回复过该用户）", False
            # 仅关键词模式：非关键词一律不回复
            if self.config.get("only_reply_on_keyword", False):
                return False, "未命中关键词（仅关键词模式）", False

            # 冷场/新话题：按概率决定是否主动搭话
            prob = float(self.config.get("reply_probability", 1.0) or 0.0)
            if prob <= 0:
                return False, "冷场概率为 0（不主动搭话）", False
            if prob < 1.0 and random.random() > prob:
                return False, f"冷场概率未命中（{prob:.0%}），不主动搭话", False
            return True, "群聊自主回复（冷场搭话）", False

        # 4. 私聊（无 @ 概念，统一视为自主回复；私聊不参与冷场概率）
        if not self.config.get("private_chat_auto_reply", True):
            return False, "私聊自主回复开关关闭", False
        if is_quiet and not self.config.get("quiet_auto_reply", False):
            return False, "免打扰时段自主回复开关关闭", True

        if hit_keyword:
            return True, "命中关键词", False
        if self.config.get("only_reply_on_keyword", False) and not hit_keyword:
            return False, "未命中关键词（当前为仅关键词模式）", False
        return True, "私聊自主回复", False

    def _hit_keyword(self, text: str) -> bool:
        """关键词匹配（不区分大小写的子串匹配）。"""
        keywords = self.config.get("keyword_triggers", []) or []
        low = text.lower()
        for kw in keywords:
            kw = str(kw or "").strip()
            if kw and kw.lower() in low:
                return True
        return False

    def _in_quiet_period(self, now: dtime) -> bool:
        """判断当前时间是否处于任一免打扰时段。时段格式 "22:00-9:00"，支持跨天。"""
        periods = self.config.get("quiet_periods", []) or []
        for p in periods:
            start, end = self._parse_period(p)
            if start is None or end is None:
                continue
            if start == end:
                continue  # 起止相同视为无效时段
            if start < end:
                if start <= now < end:
                    return True
            else:  # 跨天，如 22:00 - 7:00
                if now >= start or now < end:
                    return True
        return False

    @staticmethod
    def _parse_period(p: Any) -> tuple[Optional[dtime], Optional[dtime]]:
        """解析 "22:00-9:00" 形式的时段字符串（支持 -、~、— 分隔，容忍空格）；
        兼容旧版配置残留的 dict 格式 {"start":"22:00","end":"9:00"}。"""
        if isinstance(p, dict):
            return (
                ProactiveReplyPlugin._parse_time(p.get("start")),
                ProactiveReplyPlugin._parse_time(p.get("end")),
            )
        s = str(p or "").strip().replace("：", ":")
        for sep in ("~", "—", "–", "-"):
            if sep in s:
                parts = s.split(sep)
                if len(parts) == 2:
                    return (
                        ProactiveReplyPlugin._parse_time(parts[0]),
                        ProactiveReplyPlugin._parse_time(parts[1]),
                    )
        return None, None

    @staticmethod
    def _parse_time(s: Any) -> Optional[dtime]:
        try:
            hh, mm = str(s or "").strip().split(":")
            return dtime(int(hh), int(mm))
        except Exception:
            return None

    def _is_at_me(self, event: AstrMessageEvent) -> bool:
        """判断消息是否 @ 了机器人（或带唤醒前缀），多级兜底，兼容不同平台/版本。"""
        # 1) 优先使用 AstrBot 官方方法（部分版本存在 is_at()）
        is_at = getattr(event, "is_at", None)
        if callable(is_at):
            try:
                return bool(is_at())
            except Exception:
                pass

        # 2) 手动扫描消息链中的 At 段（aiocqhttp 等平台）
        msg_obj = event.message_obj
        self_id = getattr(msg_obj, "self_id", None)
        chain = getattr(msg_obj, "message", None) or []
        for comp in chain:
            if isinstance(comp, At):
                qq = getattr(comp, "qq", None)
                if qq is None:
                    continue
                qq_str = str(qq)
                # 群 @ 全体等场景
                if qq_str in ("all", "0", "-1"):
                    return True
                if self_id and qq_str == str(self_id):
                    return True

        # 3) 平台属性兜底（QQ 官方接口等不支持 At 段的平台）：
        #    is_at_or_wake_command = 被 @ 或带唤醒前缀（或私聊）。
        #    该方法仅在群聊流程中被使用，私聊不会走到这里。
        try:
            if getattr(event, "is_at_or_wake_command", False):
                return True
        except Exception:
            pass
        return False

    def _has_at_other(self, event: AstrMessageEvent) -> bool:
        """判断群聊消息里是否 @ 了别人（不是 @ 机器人自己）。

        这种消息是群友之间在对话，bot 不应插嘴。
        """
        msg_obj = event.message_obj
        self_id = str(getattr(msg_obj, "self_id", "") or "")
        chain = getattr(msg_obj, "message", None) or []
        for comp in chain:
            if isinstance(comp, At):
                qq = str(getattr(comp, "qq", "") or "")
                if not qq or qq in ("all", "0", "-1"):
                    continue
                if qq != self_id:
                    return True
        return False

    # ------------------------------------------------------------------ #
    # 会话记忆
    # ------------------------------------------------------------------ #
    def _get_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _get_history(self, key: str) -> deque:
        hist = self._histories.get(key)
        if hist is None:
            max_rounds = int(self.config.get("max_history_rounds", 20) or 20)
            hist = deque(maxlen=max(1, max_rounds) * 2)  # 每轮一问一答共 2 条
            self._histories[key] = hist
        return hist

    @staticmethod
    def _build_messages(history: deque, prompt_text: str) -> list[dict]:
        """将记忆 + 当前消息组装成给 LLM 的 messages。"""
        msgs = list(history)
        msgs.append({"role": "user", "content": prompt_text})
        return msgs

    def _has_pending_context(self, umo: str, sender_name: str) -> bool:
        """检测「延续信号」：机器人刚刚回复过这位发送者。

        群聊历史中每条用户消息带 "发送者: 内容" 前缀，若历史最后两条为
        [user(来自 sender_name), assistant(机器人回复)]，说明机器人刚回复过此人，
        此人紧接着说话视为对话延续，应直接回复，不再让模型判断是否无关。
        """
        hist = self._get_history(umo)
        if len(hist) < 2:
            return False
        if hist[-1].get("role") != "assistant":
            return False
        last_user = hist[-2]
        if last_user.get("role") != "user":
            return False
        content = str(last_user.get("content", ""))
        if ":" not in content:
            return False
        name = content.split(":", 1)[0].strip()
        return bool(name) and name == sender_name

    def _is_recent_duplicate(self, umo: str, text: str, window_sec: float = DUPLICATE_WINDOW_SEC) -> bool:
        """同一会话内，窗口期内是否已处理过完全相同的文本。"""
        recent = self._recent_msgs.get(umo)
        if not recent:
            return False
        now = time.time()
        for t, ts in recent:
            if t == text and now - ts <= window_sec:
                return True
        return False

    def _remember_msg(self, umo: str, text: str) -> None:
        """记录一条已处理的消息，用于防重复回复。"""
        recent = self._recent_msgs.setdefault(umo, deque(maxlen=20))
        recent.append((text, time.time()))

    # ------------------------------------------------------------------ #
    # LLM 调用（新旧两套 API 兼容；v1.7.0 起接通 on_llm_request/response 钩子链）
    # ------------------------------------------------------------------ #
    async def _llm_chat(
        self,
        event: AstrMessageEvent,
        umo: str,
        messages: list[dict],
        need_context_judge: bool = False,
    ) -> tuple[Optional[str], str]:
        """调用 AstrBot 的 LLM 能力生成回复。

        历史消息以明文拼入 prompt（[之前的对话]...），不依赖 provider 的
        contexts 参数，确保任何提供商都能真正收到上下文（修复“聊久了不接话”）。

        v1.7.0：在调用 provider 前后手动触发 on_llm_request / on_llm_response
        钩子，让 lili_state 等通过钩子注入人设/维护状态的插件正常生效。

        返回 (回复文本, 状态)。状态取值：
        - "ok"：正常回复；
        - "no_reply"：上下文延续判断为「与机器人无关/重复话题/判定确认文本」；
        - "empty"：模型返回了空文本；
        - "failed"：LLM 调用失败或未找到提供商。
        """
        prov = None

        # 1) 用户指定了 provider_id
        provider_id = str(self.config.get("provider_id", "") or "").strip()
        if provider_id:
            try:
                getter = getattr(self.context, "get_provider_by_id", None)
                if callable(getter):
                    prov = getter(provider_id)
                    self._log("INFO", f"使用指定提供商 ID：{provider_id}")
            except Exception as e:
                self._log("WARNING", f"按 ID 获取提供商失败: {e!r}")

        # 2) 使用会话绑定的提供商（默认）
        if prov is None and self.config.get("use_session_provider", True):
            getter_async = getattr(self.context, "get_using_provider_async", None)
            if callable(getter_async):
                try:
                    prov = await getter_async(umo=umo)
                except Exception as e:
                    self._log("DEBUG", f"get_using_provider_async 失败: {e!r}")
            if prov is None:
                getter_sync = getattr(self.context, "get_using_provider", None)
                if callable(getter_sync):
                    try:
                        prov = getter_sync()
                    except Exception as e:
                        self._log("DEBUG", f"get_using_provider 失败: {e!r}")

        if prov is None:
            self._log("ERROR", "未找到可用的 LLM 提供商，请检查模型配置")
            return None, "failed"

        # 明文历史 + 当前消息拼成 prompt（历史一定进入模型）
        history_msgs = messages[:-1] if len(messages) > 1 else []
        current_text = messages[-1]["content"] if messages else ""
        prompt_text = self._format_context_prompt(history_msgs, current_text)
        if need_context_judge:
            # 使用配置面板「上下文判断提示词」（可自定义）；未设置时用默认值
            judge = str(self.config.get("context_judge_prompt", "") or "").strip()
            if not judge:
                judge = DEFAULT_CONTEXT_JUDGE_INSTRUCTION
            prompt_text = prompt_text + judge

        # ---- v1.7.4：触发 on_llm_request 钩子链，注入人设/状态 ----
        # 注意点：
        # 1. call_event_hook 内部会用 event.plugins_name 做插件白名单过滤，
        #    从本插件内部触发时 event.plugins_name 通常只包含当前插件名，
        #    会把 lili_state / master_aware 等插件的 handler 过滤掉，因此临时放开为 ["*"]。
        # 2. 各插件注入方式不同：
        #    - lili_state 直接改写 req.prompt（人设前缀拼在用户消息前）
        #    - master_aware 往 req.extra_user_content_parts 追加临时内容块
        #    - 其他插件可能改 req.system_prompt / req.contexts 等
        #    因此钩子跑完后必须把整个 req 对象传给 provider，不能只取 prompt/system_prompt。
        # 3. v1.7.4：加载 AstrBot 原生 conversation 历史到 req.contexts，
        #    这样 bot 能记得本插件安装之前就存在的对话上下文。
        final_req = None
        hook_stopped = False
        _old_plugins_name = None
        if _HOOKS_AVAILABLE and event is not None:
            try:
                _old_plugins_name = getattr(event, "plugins_name", None)
                try:
                    event.plugins_name = ["*"]
                except Exception:
                    pass

                req = ProviderRequest(
                    prompt=prompt_text,
                    system_prompt="",
                    session_id=umo or "",
                )

                # 加载 AstrBot 原生会话历史（包含插件安装前的对话）
                try:
                    conv_mgr = getattr(self.context, "conversation_manager", None)
                    if conv_mgr is not None and umo:
                        cid = await conv_mgr.get_curr_conversation_id(umo)
                        if not cid:
                            cid = await conv_mgr.new_conversation(
                                umo, event.get_platform_id()
                            )
                        conv = await conv_mgr.get_conversation(umo, cid)
                        if conv is not None:
                            req.conversation = conv
                            import json as _json
                            req.contexts = _json.loads(conv.history or "[]")
                            self._log(
                                "INFO",
                                f"已加载原生会话历史：{len(req.contexts)} 条消息",
                            )
                except Exception as e:
                    self._log("DEBUG", f"加载原生会话历史失败（不影响回复）：{e!r}")

                stopped = await call_event_hook(
                    event, EventType.OnLLMRequestEvent, req
                )
                if stopped:
                    self._log("INFO", "on_llm_request 钩子要求终止本次请求")
                    hook_stopped = True
                else:
                    final_req = req
                    self._log(
                        "INFO",
                        f"on_llm_request 钩子执行完毕：prompt {len(req.prompt or '')} 字"
                        + f"，system_prompt {len(req.system_prompt or '')} 字"
                        + f"，extra_parts {len(req.extra_user_content_parts or [])} 个"
                        + f"，contexts {len(req.contexts or [])} 条",
                    )
            except Exception as e:
                # 钩子异常不能阻断回复主流程，降级为不带人设继续
                self._log("WARNING", f"触发 on_llm_request 钩子异常，已跳过：{e!r}")
                final_req = None
            finally:
                try:
                    event.plugins_name = _old_plugins_name
                except Exception:
                    pass

        if hook_stopped:
            return None, "no_reply"

        # 旧版 text_chat 路径
        try:
            resp = await asyncio.wait_for(
                self._call_text_chat(prov, prompt_text, final_req=final_req),
                timeout=120,
            )
            text = self._extract_completion(resp)

            # ---- v1.7.1：触发 on_llm_response 钩子链，更新人设插件状态 ----
            if _HOOKS_AVAILABLE and event is not None:
                try:
                    try:
                        event.plugins_name = ["*"]
                    except Exception:
                        pass
                    fake_resp = LLMResponse(role="assistant", completion_text=text or "")
                    await call_event_hook(
                        event, EventType.OnLLMResponseEvent, fake_resp
                    )
                except Exception as e:
                    self._log("WARNING", f"触发 on_llm_response 钩子异常，已跳过：{e!r}")
                finally:
                    try:
                        event.plugins_name = _old_plugins_name
                    except Exception:
                        pass

            if not text:
                self._log("WARNING", f"text_chat 未返回文本: role={getattr(resp, 'role', '?')}")
                return None, "empty"
            # 任何路径下都识别并丢弃判定确认文本（判断模式与非判断模式均适用）
            if self._is_no_reply(text):
                return None, "no_reply"
            if self._looks_like_judge_output(text):
                self._log("WARNING", f"模型输出了判定确认文本而非回复，已丢弃 | {text[:60]}")
                return None, "no_reply"
            return text, "ok"
        except asyncio.TimeoutError:
            self._log("ERROR", "LLM 调用超时（120s）")
        except Exception as e:
            self._log("WARNING", f"text_chat 调用失败: {e!r}")

        # 新版 llm_generate 路径（v4.5.7+）
        try:
            result = await self._llm_generate_new(umo, prompt_text, need_context_judge)
            if result is not None:
                return result
        except Exception as e:
            self._log("WARNING", f"llm_generate 调用异常: {e!r}")
        return None, "failed"

    @staticmethod
    def _format_context_prompt(history_msgs: list[dict], current_text: str) -> str:
        """把最近历史 + 当前消息拼成明文 prompt，确保历史一定传给模型。"""
        lines = []
        for m in history_msgs:
            role = m.get("role")
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                lines.append("机器人: " + content)
            else:
                # user 消息在群聊中已带“发送者: ”前缀
                lines.append(content)
        parts = []
        if lines:
            parts.append("[之前的对话]\n" + "\n".join(lines))
        parts.append("[当前消息]\n" + current_text)
        return "\n\n".join(parts)

    async def _call_text_chat(
        self, prov: Any, fallback_prompt: str, final_req: Any = None
    ) -> Any:
        """按实际签名调用 provider.text_chat()。

        v1.7.3：如果钩子链注入了 final_req（被 lili_state/master_aware 等插件
        改写过的 ProviderRequest），则把 req 上所有可能被插件修改的字段
        （prompt / system_prompt / extra_user_content_parts / session_id /
        contexts / image_urls / audio_urls）完整传给 provider，确保所有
        人设/认主插件的注入都生效，本插件只负责"自主回复决策"，不干涉内容。
        """
        # 钩子没跑或失败时，退回裸调（仅传 prompt）
        if final_req is None:
            try:
                return await prov.text_chat(prompt=fallback_prompt)
            except TypeError:
                return await prov.text_chat(fallback_prompt)

        # 从被插件改写过的 req 里取所有字段
        kwargs: dict[str, Any] = {
            "prompt": final_req.prompt or fallback_prompt,
            "system_prompt": (final_req.system_prompt or None),
            "session_id": final_req.session_id or None,
        }
        if final_req.extra_user_content_parts:
            kwargs["extra_user_content_parts"] = final_req.extra_user_content_parts
        if final_req.contexts:
            kwargs["contexts"] = final_req.contexts
        if final_req.image_urls:
            kwargs["image_urls"] = final_req.image_urls
        if final_req.audio_urls:
            kwargs["audio_urls"] = final_req.audio_urls

        try:
            return await prov.text_chat(**kwargs)
        except TypeError:
            # 个别 provider 不接受部分关键字参数，逐步剔除重试
            for drop_key in (
                "extra_user_content_parts",
                "contexts",
                "audio_urls",
                "image_urls",
                "session_id",
                "system_prompt",
            ):
                kwargs.pop(drop_key, None)
                try:
                    return await prov.text_chat(**kwargs)
                except TypeError:
                    continue
            # 最终兜底：只传 prompt
            return await prov.text_chat(prompt=final_req.prompt or fallback_prompt)

    async def _llm_generate_new(
        self, umo: str, prompt: str, need_context_judge: bool
    ) -> Optional[tuple[str, str]]:
        """新版 AstrBot (>= v4.5.7) 的 llm_generate() 调用。返回 (回复文本, 状态)。"""
        if not (hasattr(self.context, "llm_generate") and hasattr(self.context, "get_current_chat_provider_id")):
            return None
        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            if not provider_id:
                self._log("WARNING", "llm_generate: 未获取到会话绑定的提供商 ID")
                return None
            try:
                resp = await asyncio.wait_for(
                    self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=prompt,
                    ),
                    timeout=120,
                )
            except TypeError:
                resp = await asyncio.wait_for(
                    self.context.llm_generate(provider_id, prompt),
                    timeout=120,
                )
            text = self._extract_completion(resp)
            if not text:
                self._log("WARNING", "llm_generate 未返回文本")
                return ("", "empty")
            # 任何路径下都识别并丢弃判定确认文本（判断模式与非判断模式均适用）
            if self._is_no_reply(text):
                return (None, "no_reply")
            if self._looks_like_judge_output(text):
                self._log("WARNING", f"模型输出了判定确认文本而非回复，已丢弃 | {text[:60]}")
                return (None, "no_reply")
            return (text, "ok")
        except asyncio.TimeoutError:
            self._log("ERROR", "llm_generate 调用超时（120s）")
        except Exception as e:
            self._log("WARNING", f"llm_generate 调用失败: {e!r}")
        return None

    @staticmethod
    def _is_no_reply(text: str) -> bool:
        """识别模型输出的「不回复」标记。"""
        t = (text or "").strip()
        if "__NO_REPLY__" in t:
            return True
        # 兼容模型只输出 NO_REPLY / NO 的情况
        if re.fullmatch(r"no[\s_]*reply[.!]*", t, re.IGNORECASE):
            return True
        if re.fullmatch(r"no[.!]*", t, re.IGNORECASE) and len(t) <= 3:
            return True
        return False

    @staticmethod
    def _looks_like_judge_output(text: str) -> bool:
        """识别「判定确认文本」：模型没有给出真实回复，而是回答了判断结论
        （如“是，这条消息是在与我继续对话。”）。识别到则丢弃，防止误发。"""
        t = (text or "").strip()
        if not t:
            return False
        for pat in _JUDGE_TEXT_PATTERNS:
            if pat.search(t):
                return True
        return False

    @staticmethod
    def _extract_completion(resp: Any) -> str:
        """从 LLMResponse 提取回复文本。"""
        if resp is None:
            return ""
        role = getattr(resp, "role", "")
        if role == "err":
            return ""
        text = getattr(resp, "completion_text", None)
        if text:
            return str(text).strip()
        chain = getattr(resp, "result_chain", None)
        if chain is not None:
            try:
                plain = chain.get_plain_text()
                if plain:
                    return str(plain).strip()
            except Exception:
                pass
        return ""

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_get(event: AstrMessageEvent, method: str, default: str) -> str:
        fn = getattr(event, method, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                pass
        return default

    # ------------------------------------------------------------------ #
    # 管理指令
    # ------------------------------------------------------------------ #
    @filter.command("autoreply_status")
    async def autoreply_status(self, event: AstrMessageEvent):
        """查看智能自主回复插件的当前状态与配置摘要"""
        g = lambda k, d: self.config.get(k, d)
        lines = [
            "【智能自主回复 · 状态】",
            f"插件总开关: {'开' if g('enable_plugin', True) else '关'}",
            f"私聊自主回复: {'开' if g('private_chat_auto_reply', True) else '关'}",
            f"群聊自主回复(非@): {'开' if g('group_chat_auto_reply', False) else '关'}",
            f"上下文延续判断: {'开' if g('context_aware_reply', True) else '关'}",
            f"自主回复概率: {g('reply_probability', 1.0)}",
            f"仅关键词模式: {'是' if g('only_reply_on_keyword', False) else '否'}",
            f"关键词: {', '.join(map(str, g('keyword_triggers', []) or [])) or '(未设置)'}",
            f"免打扰: {'启用' if g('quiet_periods_enabled', True) else '关闭'} | "
            f"{self._format_periods(g('quiet_periods', []) or [])}",
            f"免打扰内自主回复: {'开' if g('quiet_auto_reply', False) else '关'} | "
            f"免打扰内@回复: {'开' if g('quiet_at_reply', False) else '关'}",
            f"记忆轮数: {g('max_history_rounds', 20)}",
            f"日志文件: {LOG_FILE}",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("autoreply_log")
    async def autoreply_log(self, event: AstrMessageEvent, lines: int = 30):
        """查看插件运行日志（默认最近 30 行，可指定行数，如 /autoreply_log 100）"""
        try:
            n = max(1, min(int(lines), 200))
        except Exception:
            n = 30
        if not LOG_FILE.exists():
            yield event.plain_result("日志文件尚未生成（可能还没有消息经过插件）。")
            return
        try:
            content = LOG_FILE.read_text(encoding="utf-8", errors="replace")
            tail = content.splitlines()[-n:]
            yield event.plain_result(f"【插件日志 · 最近 {len(tail)} 行】\n" + "\n".join(tail))
        except Exception as e:
            yield event.plain_result(f"读取日志失败: {e!r}")

    @staticmethod
    def _format_periods(periods: list) -> str:
        if not periods:
            return "(未设置)"
        parts = [str(p).strip() for p in periods if str(p or "").strip()]
        return "; ".join(parts) or "(未设置)"

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    async def terminate(self):
        """插件卸载/停用时：取消主动问候后台任务 + 清理日志 handler。"""
        # 1) 取消私聊主动问候后台任务
        try:
            if self._greeting_task is not None:
                self._greeting_task.cancel()
                self._greeting_task = None
                self._log("INFO", "私聊主动问候后台任务已停止")
        except Exception:
            pass
        # 2) 清理文件日志 handler
        try:
            if self._file_logger is not None:
                for h in list(self._file_logger.handlers):
                    try:
                        h.close()
                    except Exception:
                        pass
                self._file_logger.handlers.clear()
        except Exception:
            pass
