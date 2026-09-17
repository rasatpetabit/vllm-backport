# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming parser engine that orchestrates token ID scanning,
incremental lexing, and state-machine-driven semantic event emission."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.incremental_lexer import (
    CONTENT_TERMINAL,
    IncrementalLexer,
    LexerShape,
    LexToken,
    TerminalDef,
)
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.token_id_scanner import (
    DROP_TERMINAL,
    LexerInput,
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)


@dataclass(slots=True)
class _DropInfo:
    lexer_shape: LexerShape
    extra_token_ids: dict[int, str]


def _build_drop_info(
    config: ParserEngineConfig,
    tokenizer,
) -> _DropInfo | None:
    try:
        special_tokens: list[str] = list(tokenizer.all_special_tokens)
        special_ids: list[int] = list(tokenizer.all_special_ids)
    except (AttributeError, NotImplementedError):
        return None

    if not special_tokens:
        return None

    configured_texts = (
        set(config.token_id_terminals.values())
        | config.terminal_literals
        | config.preserve_tokens
    )

    extra_token_ids: dict[int, str] = {}
    drop_texts: set[str] = set()
    for text, tid in zip(special_tokens, special_ids):
        if text not in configured_texts:
            extra_token_ids[tid] = DROP_TERMINAL
            drop_texts.add(text)

    if not drop_texts:
        return None

    import regex as re

    drop_terminal_defs = [
        TerminalDef(
            name=DROP_TERMINAL,
            pattern=re.compile(re.escape(text)),
            is_literal=True,
            literal=text,
        )
        for text in drop_texts
    ]

    all_terminal_defs = list(config.terminal_defs) + drop_terminal_defs
    lexer_shape = LexerShape(all_terminal_defs)

    return _DropInfo(
        lexer_shape=lexer_shape,
        extra_token_ids=extra_token_ids,
    )


class StreamingParserEngine:
    """Consumes ``(delta_text, delta_token_ids)`` pairs and produces a
    stream of :class:`SemanticEvent` instances.

    This is the main entry point for streaming parsing.
    Create one per request (it is stateful).

    The pipeline is::

        delta_text + delta_token_ids
            → TokenIDScanner  (special token pre-lexing)
            → IncrementalLexer  (text → terminal tokens with prefix buffering)
            → State Machine  (terminal → semantic events)
            → list[SemanticEvent]

    Usage::

        engine = StreamingParserEngine(config, tokenizer)
        for each streaming delta:
            events = engine.feed(delta_text, delta_token_ids)
            # convert events to DeltaMessage
    """

    def __init__(
        self,
        config: ParserEngineConfig,
        tokenizer,
        initial_state: ParserState | None = None,
        vocab: dict[str, int] | None = None,
    ) -> None:
        self.config = config

        resolved_token_ids: dict[int, str] = {}
        if tokenizer is not None:
            if vocab is None:
                vocab = tokenizer.get_vocab()
            if config.token_id_terminals:
                for terminal_name, token_text in config.token_id_terminals.items():
                    tid = vocab.get(token_text)
                    if tid is not None:
                        resolved_token_ids[tid] = terminal_name

        drop_info: _DropInfo | None = None
        if tokenizer is not None:
            drop_info = _build_drop_info(config, tokenizer)

        lexer_shape = config.lexer_shape
        if drop_info is not None:
            resolved_token_ids.update(drop_info.extra_token_ids)
            lexer_shape = drop_info.lexer_shape

        self._resolved_token_ids = resolved_token_ids
        self._has_drops = drop_info is not None

        self._scanner = TokenIDScanner(
            resolved_token_ids,
            tokenizer,
        )

        self._token_id_terminal_names: frozenset[str] = frozenset(
            resolved_token_ids.values()
        )

        self._lexer = IncrementalLexer(lexer_shape, content_terminal=CONTENT_TERMINAL)

        self._tool_terminals: frozenset[str] = frozenset(
            terminal
            for (state, terminal), tr in config.transitions.items()
            if tr.next_state in self._TOOL_STATES or state in self._TOOL_STATES
        )
        # TOOL_CALL_END may close an inner call rather than its lexical wrapper,
        # as in MiniMax, so identify exits from state transitions instead.
        self._tool_exit_terminals: frozenset[str] = frozenset(
            terminal
            for (state, terminal), tr in config.transitions.items()
            if state in self._TOOL_STATES and tr.next_state not in self._TOOL_STATES
        )

        self._reasoning_markup_terminals: frozenset[str] = (
            self._compute_reasoning_markup_terminals()
        )

        self.skip_tool_parsing = False
        self.skip_reasoning_parsing = False
        # Function names declared by the request, or None when unknown.
        # Consulted only by transitions with ``validate_tool_name``;
        # set per request by the owning ParserEngine, like
        # ``skip_tool_parsing`` it survives reset().
        self.allowed_tool_names: frozenset[str] | None = None
        # True when the request asked for tool_choice "none".  Recovery
        # transitions are skipped while set, so text that looks like a
        # recovered tool call stays plain content instead of being
        # consumed and then suppressed.  Set per request by the owning
        # ParserEngine; survives reset() like ``skip_tool_parsing``.
        self.suppress_tool_calls = False
        # Required parameter names per declared tool, or None when unknown.
        # Consulted only when deciding whether to commit a *recovered*
        # tool call, which only happens under strict admission.  Set per
        # request by the owning ParserEngine alongside
        # ``allowed_tool_names``; survives reset() the same way.
        self.required_tool_params: dict[str, frozenset[str]] | None = None
        self.reset(initial_state=initial_state)

    @property
    def reasoning_token_count(self) -> int:
        return self._reasoning_token_count

    def _record_reasoning_tokens(self, events: Sequence[SemanticEvent]) -> None:
        self._reasoning_token_count += sum(
            event.token_count
            for event in events
            if event.type == EventType.REASONING_CHUNK
        )

    def _reset_args_state(self) -> None:
        self._args_buffer: str = ""
        self._args_safe_end: int = 0
        self._args_brace_depth: int = 0
        self._args_in_string: bool = False
        self._args_escape_next: bool = False

    def reset(self, initial_state: ParserState | None = None) -> None:
        """Reset mutable state for reuse across requests.

        Preserves cached immutable structures (compiled terminals,
        resolved token IDs, lexer shape, token text cache) to avoid
        redundant initialization work.
        """
        self.state = (
            initial_state if initial_state is not None else self.config.initial_state
        )
        self.tool_index = -1
        self._ever_had_token_ids = False
        self._reasoning_token_count = 0
        # DO NOT reset skip_tool_parsing here — callers set it before
        # calling methods that trigger reset() (e.g. extract_reasoning),
        # and clearing it silently breaks non-streaming tool-call-as-
        # implicit-reasoning-end (content returns None).
        self._scanner.reset()
        self._lexer.reset()
        self._message_header_buffer = ""
        self._message_header_token_count = 0
        self._in_skipped_tool_span = False
        self._reset_args_state()
        self._recovered_tool_call = False
        self._pending_between_text = ""
        self._hold_active = False
        # "name" while the recovered tool name is still being read,
        # "body" once it validated and the block must prove it closes.
        # Advanced only on the strict_tool_call_admission path.
        self._hold_phase = "name"
        self._held_events: list[SemanticEvent] = []
        self._held_raw: list[str] = []
        self._held_name: list[str] = []
        self._held_prior_state: ParserState = self.state
        self._held_prior_tool_index: int = -1

    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        if delta_token_ids:
            self._ever_had_token_ids = True

        # Fast path: skip scanner and lexer when the delta is plain
        # content with no special tokens and no terminal-starting chars.
        if (
            delta_text
            and not self._lexer.buffer
            and not self._scanner._deferred_terminals
            and self._lexer._literal_first_chars.isdisjoint(delta_text)
        ):
            has_special = False
            for tid in delta_token_ids:
                if tid in self._resolved_token_ids:
                    has_special = True
                    break
            if not has_special:
                events = self._emit_for_state(
                    delta_text, token_count=len(delta_token_ids)
                )
                self._record_reasoning_tokens(events)
                return events

        scanner_items = self._scanner.scan(delta_text, delta_token_ids)

        if len(scanner_items) == 1 and isinstance(scanner_items[0], TextChunk):
            item = scanner_items[0]
            lex_tokens = self._lexer.feed(item.text, item.token_texts, item.token_count)
            if len(lex_tokens) == 1 and lex_tokens[0].terminal == CONTENT_TERMINAL:
                events = self._emit_for_state(
                    lex_tokens[0].value,
                    token_count=lex_tokens[0].token_count,
                )
            else:
                events = self._process_lex_tokens(lex_tokens)
            self._record_reasoning_tokens(events)
            return events

        events = self._process_scanner_items(scanner_items)
        self._record_reasoning_tokens(events)
        return events

    def _process_scanner_items(
        self, items: Sequence[LexerInput]
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        for item in items:
            if isinstance(item, PreLexedTerminal):
                events.extend(self._process_lex_tokens(self._lexer.flush()))
                events.extend(self._on_terminal(item.terminal, item.text))
            elif isinstance(item, TextChunk):
                if not item.text and item.token_count:
                    events.extend(
                        self._emit_for_state("", token_count=item.token_count)
                    )
                else:
                    events.extend(
                        self._process_lex_tokens(
                            self._lexer.feed(
                                item.text, item.token_texts, item.token_count
                            )
                        )
                    )
        return events

    def finish(self) -> list[SemanticEvent]:
        events = self._process_scanner_items(self._scanner.flush_pending())

        events.extend(self._process_lex_tokens(self._lexer.flush()))

        if self._hold_active:
            # Stream ended before the recovered tool name completed:
            # the held events never validated, so flush the raw text
            # as content in the pre-recovery state.
            events.extend(self._abort_hold("".join(self._held_raw)))

        if self._args_buffer:
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""
            self._args_safe_end = 0

        if self.state in (
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_ARGS,
            ParserState.TOOL_NAME,
            ParserState.TOOL_BETWEEN,
        ):
            if self.tool_index >= 0:
                events.append(
                    SemanticEvent(
                        EventType.TOOL_CALL_END,
                        tool_index=self.tool_index,
                    )
                )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.REASONING:
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.MESSAGE_HEADER:
            if self._message_header_buffer:
                events.append(
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=self._message_header_buffer,
                        tool_index=self.tool_index,
                        token_count=self._message_header_token_count,
                    )
                )
                self._message_header_buffer = ""
                self._message_header_token_count = 0
            self.state = ParserState.CONTENT

        self._record_reasoning_tokens(events)
        return events

    def parse_complete(self, text: str) -> list[SemanticEvent]:
        token_ids: list[int] = []
        events = self.feed(text, token_ids)
        events.extend(self.finish())
        return events

    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        strict = self._token_id_terminal_names if self._ever_had_token_ids else None
        for tok in tokens:
            if tok.terminal == CONTENT_TERMINAL or (strict and tok.terminal in strict):
                events.extend(self._on_content(tok.value, tok.token_count))
            else:
                events.extend(
                    self._on_terminal(tok.terminal, tok.value, tok.token_count)
                )
        return events

    _TOOL_STATES = frozenset(
        {
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_NAME,
            ParserState.TOOL_ARGS,
            ParserState.TOOL_BETWEEN,
        }
    )

    _PLAIN_STATES = frozenset({ParserState.CONTENT, ParserState.REASONING})

    _REASONING_EVENTS = frozenset({EventType.REASONING_START, EventType.REASONING_END})

    def _compute_reasoning_markup_terminals(self) -> frozenset[str]:
        """Terminals the ``skip_reasoning_parsing`` bypass may neutralize.

        Only reasoning-exclusive markers qualify: every transition they
        participate in stays within CONTENT/REASONING and emits nothing
        but reasoning events. Inkling's ``<|end_message|>`` is labelled
        THINK_END yet also closes text, header, and tool blocks;
        bypassing a shared marker would eat that structure, so one impure
        marker disables the bypass for the whole config.
        """
        markers = frozenset(
            terminal
            for (state, terminal), tr in self.config.transitions.items()
            if ParserState.REASONING in (state, tr.next_state)
            and tr.next_state not in self._TOOL_STATES
        )
        for (state, terminal), tr in self.config.transitions.items():
            if terminal not in markers:
                continue
            if (
                state not in self._PLAIN_STATES
                or tr.next_state not in self._PLAIN_STATES
                or not self._REASONING_EVENTS.issuperset(tr.events)
            ):
                return frozenset()
        return markers

    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        key = (self.state, terminal)
        transition = self.config.transitions.get(key)

        if transition is None:
            if self._has_drops and terminal == DROP_TERMINAL:
                return []
            if self._hold_active and self.state == ParserState.TOOL_NAME:
                # A terminal with no meaning inside a held tool name,
                # for example a real tool call start token, ends the
                # hold: replay the held text as content, then handle
                # the terminal again in the restored state so it keeps
                # its normal meaning.
                events = self._abort_hold("".join(self._held_raw))
                events.extend(self._on_terminal(terminal, value, token_count))
                return events
            # The projected skip state may not define the wrapper closer.
            if self.skip_tool_parsing and terminal in self._tool_exit_terminals:
                self._in_skipped_tool_span = False
            return self._emit_for_state(value, token_count)

        if self.skip_reasoning_parsing and terminal in self._reasoning_markup_terminals:
            return self._emit_for_state(value, token_count)

        # Under strict admission an active suppression means this request
        # can never yield a tool call, so tool terminals are just text the
        # model wrote; without the flag, suppression diverts recovery only
        # and terminals keep their normal transitions.
        suppress_as_content = (
            self.config.strict_tool_call_admission and self.suppress_tool_calls
        )
        if (
            self.skip_tool_parsing or suppress_as_content
        ) and terminal in self._tool_terminals:
            # Inkling reuses one terminal for tool, text, and reasoning exits.
            # Outside a forwarded tool span, apply its normal transition.
            is_opener = transition.next_state in self._TOOL_STATES
            is_exit = terminal in self._tool_exit_terminals
            used_as_plain_closer = (
                is_exit and not is_opener and not self._in_skipped_tool_span
            )
            if not used_as_plain_closer:
                if is_opener:
                    self._in_skipped_tool_span = True
                elif is_exit:
                    self._in_skipped_tool_span = False
                leaving_message_header = self.state == ParserState.MESSAGE_HEADER
                if leaving_message_header:
                    self._message_header_buffer = ""
                    self._message_header_token_count = 0
                # Reasoning ends here only when a tool call is actually
                # possible — the terminal then marks the real end of thinking.
                # When the request can never yield one the terminal is just
                # text the model wrote, often while narrating DSML syntax
                # inside <think>, so reasoning has to continue.  Ending it
                # would flush the rest of the thoughts into the content.
                # A skip_tool_parsing pass still reports REASONING_END so
                # the reasoning adapter hands the block to the tool pass.
                if (
                    not suppress_as_content
                    and EventType.REASONING_END in transition.events
                ):
                    self.state = ParserState.CONTENT
                    return [
                        SemanticEvent(
                            EventType.REASONING_END,
                            value=value,
                            tool_index=self.tool_index,
                        ),
                        SemanticEvent(
                            EventType.TEXT_CHUNK,
                            value=value,
                            tool_index=self.tool_index,
                        ),
                    ]
                elif leaving_message_header:
                    self.state = ParserState.CONTENT
                    return [
                        SemanticEvent(
                            EventType.TEXT_CHUNK,
                            value=value,
                            tool_index=self.tool_index,
                        )
                    ]
                content_type = self.config.content_events.get(self.state)
                if content_type is not None:
                    return [
                        SemanticEvent(
                            content_type, value=value, tool_index=self.tool_index
                        )
                    ]
                return []

        if transition.skip_in_token_id_mode and self._ever_had_token_ids:
            return self._emit_for_state(value, token_count)

        return self._apply_transition(transition, value, token_count)

    def _emit_for_state(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        # Body of a recovered call still awaiting its close: keep both the
        # raw text and the events it would have produced, so the block can
        # be committed whole or released as content.  Unreachable unless
        # strict admission advanced the hold into its body phase.
        if self._hold_active and self._hold_phase == "body":
            self._held_raw.append(text)
            self._hold_active = False
            try:
                self._held_events.extend(self._emit_for_state(text, token_count))
            finally:
                self._hold_active = True
            return []
        if self._hold_active and self.state == ParserState.TOOL_NAME:
            candidate = "".join(self._held_name) + text
            if not self._can_grow_into_declared_name(candidate):
                # The held text can no longer become a declared tool
                # name, so holding longer would only stall streaming.
                # Release everything consumed so far as content.
                return self._abort_hold("".join(self._held_raw) + text)
            self._held_raw.append(text)
            self._held_name.append(text)
            self._held_events.append(
                SemanticEvent(
                    EventType.TOOL_NAME,
                    value=text,
                    tool_index=self.tool_index,
                )
            )
            return []
        if self.state == ParserState.MESSAGE_HEADER:
            self._message_header_buffer += text
            self._message_header_token_count += token_count
            return []
        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)
            return [
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=text,
                    tool_index=self.tool_index,
                    token_count=token_count,
                )
            ]
        content_type = self.config.content_events.get(self.state)
        if content_type is not None:
            return [
                SemanticEvent(
                    content_type,
                    value=text,
                    tool_index=self.tool_index,
                    token_count=token_count,
                )
            ]
        if self._recovered_tool_call and self.state == ParserState.TOOL_BETWEEN:
            # A response that lost its opening wrapper usually loses the
            # closing one too, so text after a recovered invoke is often
            # the rest of the answer rather than padding before the next
            # invoke.  Whitespace is held back because that is what
            # padding looks like; as soon as anything else shows up the
            # whole run is real output and goes out as content.
            self._pending_between_text += text
            if self._pending_between_text.strip():
                held = self._pending_between_text
                self._pending_between_text = ""
                return [
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=held,
                        tool_index=self.tool_index,
                        token_count=token_count,
                    )
                ]
        return []

    def _on_content(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        if not text:
            return []
        return self._emit_for_state(text, token_count)

    def _apply_transition(
        self,
        transition: Transition,
        value: str,
        token_count: int = 0,
    ) -> list[SemanticEvent]:
        if self._hold_active:
            return self._resolve_hold(transition, value, token_count)
        if transition.validate_tool_name:
            if self.suppress_tool_calls or self.allowed_tool_names is None:
                # Recovery could never be accepted for this request, so
                # the trigger text stays plain content and nothing is
                # buffered.
                return self._emit_for_state(value, token_count)
            return self._begin_hold(transition, value, token_count)
        return self._run_transition(transition, value, token_count)

    def _begin_hold(
        self,
        transition: Transition,
        value: str,
        token_count: int = 0,
    ) -> list[SemanticEvent]:
        """Apply a ``validate_tool_name`` transition but hold its events.

        The events (and every TOOL_NAME chunk that follows) stay
        buffered until the name completes and validates, so a false
        positive can be undone without having emitted anything.
        """
        prior_state = self.state
        prior_tool_index = self.tool_index
        self._held_events = self._run_transition(transition, value, token_count)
        self._held_raw = [value]
        self._held_name = []
        self._held_prior_state = prior_state
        self._held_prior_tool_index = prior_tool_index
        self._hold_active = True
        self._hold_phase = "name"
        self._recovered_tool_call = True
        return []

    def _resolve_hold(
        self,
        transition: Transition,
        value: str,
        token_count: int = 0,
    ) -> list[SemanticEvent]:
        """Advance the hold window, ending it once the call is proven."""
        if not self.config.strict_tool_call_admission:
            name = "".join(self._held_name)
            allowed = self.allowed_tool_names
            if allowed is not None and name in allowed:
                events = self._held_events
                self._clear_hold()
                events.extend(self._run_transition(transition, value, token_count))
                return events
            return self._abort_hold("".join(self._held_raw) + value)

        # Strict admission: a declared name is not on its own enough to
        # commit a tool call that was recovered without its opening
        # wrapper — ordinary prose quoting an invoke marker carries a
        # real tool name too.  Such a block is only real if it actually
        # closes, so the hold continues through the body until a
        # TOOL_CALL_END arrives; ``finish`` releases the held text as
        # content when none ever does.
        #
        # Closing is necessary but not sufficient — prose explaining how
        # to close a block writes the closing marker too — so a recovered
        # call must also carry the arguments its tool requires.  See
        # ``_recovered_call_is_usable``.
        if self._hold_phase == "name":
            name = "".join(self._held_name)
            allowed = self.allowed_tool_names
            if allowed is None or name not in allowed:
                return self._abort_hold("".join(self._held_raw) + value)
            self._held_raw.append(value)
            self._held_events.extend(
                self._run_transition(transition, value, token_count)
            )
            self._hold_phase = "body"
            return []

        self._held_raw.append(value)
        events = self._run_transition(transition, value, token_count)
        self._held_events.extend(events)
        if any(e.type is EventType.TOOL_CALL_END for e in events):
            if not self._recovered_call_is_usable():
                return self._abort_hold("".join(self._held_raw))
            held = self._held_events
            self._clear_hold()
            return held
        return []

    def _recovered_call_is_usable(self) -> bool:
        """Whether a held, recovered call carries the arguments it needs.

        A recovered call is a guess: the invoke marker is ordinary text,
        so prose quoting a declared tool is indistinguishable from a real
        invoke at the token level.  What does separate them is the
        payload — a genuine call supplies the tool's required parameters,
        prose supplies a sentence.  Rejecting here re-emits the whole held
        span as content, which is what the text was to begin with.

        A tool that requires nothing stays ambiguous by construction and
        is accepted, as is the case where the request's schema is unknown.
        An engine without an arg_converter emits argument chunks that are
        already JSON, so the gate checks them directly.  Reached only on
        the strict_tool_call_admission path.
        """
        required = self.required_tool_params
        if not required:
            return True
        needed = required.get("".join(self._held_name))
        if not needed:
            return True
        converter = self.config.arg_converter
        raw_args = "".join(
            e.value
            for e in self._held_events
            if e.type is EventType.ARG_VALUE_CHUNK
        )
        try:
            provided = json.loads(
                converter(raw_args, False) if converter is not None
                else raw_args
            )
        except (TypeError, ValueError, RecursionError):
            return False
        return isinstance(provided, dict) and needed <= provided.keys()

    def _abort_hold(self, raw: str) -> list[SemanticEvent]:
        """Discard held events and re-emit the raw text as content."""
        self.state = self._held_prior_state
        self.tool_index = self._held_prior_tool_index
        self._recovered_tool_call = self._held_prior_state in self._TOOL_STATES
        self._clear_hold()
        return self._emit_for_state(raw)

    def _clear_hold(self) -> None:
        self._hold_active = False
        self._hold_phase = "name"
        self._held_events = []
        self._held_raw = []
        self._held_name = []

    def _can_grow_into_declared_name(self, candidate: str) -> bool:
        """Return True when *candidate* is a prefix of a declared tool name.

        Consulted while a recovery hold is active.  Membership in the
        declared set is the only way a held name can validate, so once
        the text seen so far stops being a prefix of any declared name
        the caller aborts the hold.  This also bounds how much text a
        hold can buffer to the length of the longest declared name.
        """
        allowed = self.allowed_tool_names
        if allowed is None:
            return False
        return any(name.startswith(candidate) for name in allowed)

    def _run_transition(
        self,
        transition: Transition,
        value: str,
        token_count: int = 0,
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        previous_state = self.state
        message_header = ""
        message_header_token_count = 0

        if (
            self.state == ParserState.TOOL_ARGS
            and transition.next_state != ParserState.TOOL_ARGS
            and self._args_buffer
        ):
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""

        # Whatever is still held between invokes is whitespace padding,
        # which the wrapped path drops too.
        self._pending_between_text = ""

        if previous_state == ParserState.MESSAGE_HEADER:
            message_header = self._message_header_buffer
            message_header_token_count = self._message_header_token_count
            self._message_header_buffer = ""
            self._message_header_token_count = 0

        # A real wrapper start (TOOL_PREAMBLE is only reachable through it)
        # ends the recovered sequence: the wrapped block that follows keeps
        # the ordinary between-invoke semantics.
        if (
            transition.next_state not in self._TOOL_STATES
            or transition.next_state == ParserState.TOOL_PREAMBLE
        ):
            self._recovered_tool_call = False

        self.state = transition.next_state

        for event_type in transition.events:
            if event_type == EventType.TOOL_CALL_START:
                self.tool_index += 1
            event_value = (
                message_header
                if previous_state == ParserState.MESSAGE_HEADER
                and event_type == EventType.TEXT_CHUNK
                else value
            )
            if event_type == EventType.TEXT_CHUNK and not event_value:
                continue
            events.append(
                SemanticEvent(
                    event_type,
                    value=event_value,
                    tool_index=self.tool_index,
                    token_count=(
                        message_header_token_count
                        if previous_state == ParserState.MESSAGE_HEADER
                        and event_type == EventType.TEXT_CHUNK
                        else token_count
                    ),
                )
            )

        if self.state == ParserState.TOOL_ARGS:
            self._args_brace_depth = 0
            self._args_in_string = False
            self._args_escape_next = False
            self._args_safe_end = 0

        return events

    def _feed_args_text(self, text: str) -> list[SemanticEvent]:
        """Feed text into the JSON argument streaming buffer.

        Streams argument characters incrementally while holding back
        closing braces/brackets that might change as more input arrives.
        """
        events: list[SemanticEvent] = []
        for ch in text:
            result = self._feed_args_char(ch)
            events.extend(result)
        return events

    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:
        self._args_buffer += ch

        if self._args_escape_next:
            self._args_escape_next = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if self._args_in_string:
            if ch == "\\":
                self._args_escape_next = True
            elif ch == '"':
                self._args_in_string = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch == '"':
            self._args_in_string = True
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("{", "["):
            self._args_brace_depth += 1
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("}", "]"):
            if self._args_brace_depth > 0:
                self._args_brace_depth -= 1
            if self._args_brace_depth == 0:
                return []
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        self._args_safe_end = len(self._args_buffer)
        return self._flush_safe_args()

    def _flush_safe_args(self) -> list[SemanticEvent]:
        """Emit buffered argument characters up to the safe-end watermark.

        Top-level closing braces are held back (safe_end not advanced)
        until confirmed safe by a subsequent character or finish().
        """
        if self._args_safe_end == 0:
            return []
        to_emit = self._args_buffer[: self._args_safe_end]
        self._args_buffer = self._args_buffer[self._args_safe_end :]
        self._args_safe_end = 0
        return [
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                value=to_emit,
                tool_index=self.tool_index,
            )
        ]
