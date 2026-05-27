from __future__ import annotations

import asyncio
import os
import sys
import threading
from contextlib import AbstractContextManager
from time import monotonic, sleep
from typing import TextIO


PASTE_PREFIX = "\x00paste\x00"
UNBRACKETED_PASTE_IDLE_S = 0.01


class Terminal(AbstractContextManager["Terminal"]):
    """Small raw terminal wrapper for tui2.

    The class uses the normal screen buffer on purpose. It only enables raw-ish
    input and bracketed paste while the app is active; transcript lines written
    before the live region remain in the host terminal scrollback.
    """

    def __init__(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._old_termios = None
        self._old_input_mode: int | None = None
        self._old_output_mode: int | None = None
        self._windows = os.name == "nt"

    def __enter__(self) -> "Terminal":
        if self._windows:
            self._enable_windows_vt()
        else:
            import termios
            import tty

            fd = self.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setraw(fd)
        self.write("\x1b[?2004h")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.write("\x1b[?2004l\x1b[0m\n")
        if not self._windows and self._old_termios is not None:
            import termios

            termios.tcsetattr(self.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
        if self._windows:
            self._restore_windows_vt()

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.stdout.flush()

    def read_key(self) -> str:
        ch = self._read_char()
        if self._windows:
            # Ctrl+C is often delivered as KeyboardInterrupt by getwch rather
            # than as the literal ETX byte. Normalize both forms for app logic.
            if ch in {"\x00", "\xe0"}:
                code = self._read_char()
                if code == "\n":
                    return "<C-ENTER>"
                if code == "M":
                    return "<RIGHT>"
                if code == "K":
                    return "<LEFT>"
                if code == "H":
                    return "<UP>"
                if code == "P":
                    return "<DOWN>"
                return "<" + code + ">"
        if ch != "\x1b":
            return self._coalesce_unbracketed_paste(ch)
        # Terminals report arrows, modified Enter and bracketed paste as CSI
        # sequences.  Bracketed paste must be returned as one key so pasted
        # newlines don't look like Enter presses to the app.
        second = self._read_char_after_escape()
        if second != "[":
            return "\x1b"
        return self._read_csi_key()

    def _read_char(self) -> str:
        if self._windows:
            import msvcrt

            return msvcrt.getwch()
        return self.stdin.read(1)

    def _read_char_after_escape(self) -> str | None:
        """Read the next escape-sequence byte without making bare Esc sticky.

        Interactive terminals may send a lone Esc key.  If no additional byte is
        ready shortly after Esc, treat it as a standalone key instead of blocking
        forever.  In tests and other in-memory streams, ``fileno()`` is usually
        unavailable, so a normal read is non-blocking and deterministic.
        """

        if self._windows:
            import msvcrt

            if msvcrt.kbhit():
                return self._read_char()
            # Tests may replace getwch with an in-memory iterator while kbhit()
            # still reflects the real console, so fall back to a direct read in
            # that case.  In normal interactive use, keep the timeout path so a
            # lone Esc key remains responsive.
            if getattr(msvcrt.getwch, "__module__", "msvcrt") != "msvcrt":
                return self._read_char()
            deadline = monotonic() + 0.03
            while monotonic() < deadline:
                if msvcrt.kbhit():
                    return self._read_char()
                sleep(0.001)
            return None

        try:
            fd = self.stdin.fileno()
        except (AttributeError, OSError):
            return self._read_char() or None
        import select

        ready, _, _ = select.select([fd], [], [], 0.03)
        if not ready:
            return None
        return self._read_char()

    def _coalesce_unbracketed_paste(self, initial: str) -> str:
        """Group a burst of plain terminal input into a synthetic paste key.

        Bracketed paste is not universally delivered on Windows terminals even
        after requesting VT input.  Without a fallback, pasted newlines arrive as
        literal Enter bytes and submit each line.  If more input is immediately
        buffered after a text-ish character, treat the whole burst as paste and
        let the app insert it as one composer edit.
        """

        if not self._can_start_unbracketed_paste(initial):
            return initial
        suffix = self._read_pending_burst()
        if not suffix:
            return initial
        return PASTE_PREFIX + self._normalize_paste_text(initial + suffix)

    @staticmethod
    def _can_start_unbracketed_paste(ch: str) -> bool:
        # Keep standalone control shortcuts as keys.  Tabs/newlines may appear
        # in pasted text, but only become paste if more bytes are already queued.
        return ch in {"\t", "\r", "\n"} or ch >= " "

    def _read_pending_burst(self) -> str:
        chars: list[str] = []
        deadline = monotonic() + UNBRACKETED_PASTE_IDLE_S
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            ch = self._read_available_char(timeout_s=remaining)
            if ch is None:
                break
            chars.append(ch)
            # Keep draining until the terminal input buffer has been idle for a
            # short interval.  This captures larger pastes without forcing the
            # app to repaint once per character.
            deadline = monotonic() + UNBRACKETED_PASTE_IDLE_S
        return "".join(chars)

    def _read_available_char(self, *, timeout_s: float) -> str | None:
        if self._windows:
            import msvcrt

            deadline = monotonic() + timeout_s
            while monotonic() < deadline:
                if msvcrt.kbhit():
                    return self._read_char()
                sleep(0.001)
            return None

        try:
            fd = self.stdin.fileno()
        except (AttributeError, OSError):
            return None
        import select

        ready, _, _ = select.select([fd], [], [], max(0.0, timeout_s))
        if not ready:
            return None
        return self._read_char()

    def _read_csi_key(self) -> str:
        sequence = "\x1b["
        while len(sequence) < 32:
            ch = self._read_char()
            if not ch:
                break
            sequence += ch
            if ch == "~" or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                break

        if sequence == "\x1b[A":
            return "<UP>"
        if sequence == "\x1b[B":
            return "<DOWN>"
        if sequence == "\x1b[C":
            return "<RIGHT>"
        if sequence == "\x1b[D":
            return "<LEFT>"
        if sequence == "\x1b[27;5;13~":
            return "<C-ENTER>"
        if sequence == "\x1b[200~":
            return PASTE_PREFIX + self._read_bracketed_paste()
        return "\x1b"

    def _read_bracketed_paste(self) -> str:
        terminator = "\x1b[201~"
        chars: list[str] = []
        while True:
            ch = self._read_char()
            if not ch:
                break
            chars.append(ch)
            if len(chars) >= len(terminator) and "".join(chars[-len(terminator) :]) == terminator:
                del chars[-len(terminator) :]
                break
        return self._normalize_paste_text("".join(chars))

    @staticmethod
    def _normalize_paste_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _enable_windows_vt(self) -> None:
        """Enable VT input/output when running under a Windows console.

        Bracketed paste only helps if the console is allowed to deliver VT input
        sequences such as ``ESC [ 200 ~``.  If the handle calls fail (for
        example under redirected stdio), fall back to the classic no-op trick
        that enables ANSI output on older consoles.
        """

        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            mode = wintypes.DWORD()
            input_handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
            if kernel32.GetConsoleMode(input_handle, ctypes.byref(mode)):
                self._old_input_mode = int(mode.value)
                # Clear ENABLE_PROCESSED_INPUT (0x0001) so Ctrl+C is delivered
                # as a raw ETX byte ("\x03") by ``msvcrt.getwch`` instead of
                # being translated by the console driver into a SIGINT signal.
                # The TUI also uses ``TerminalKeyReader`` to tolerate consoles
                # or child processes that re-enable processed input later, but
                # preferring raw ETX here avoids routing Ctrl+C through Python's
                # SIGINT machinery in the normal case.
                new_input = (mode.value | 0x0200) & ~0x0001
                kernel32.SetConsoleMode(input_handle, new_input)

            output_handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            if kernel32.GetConsoleMode(output_handle, ctypes.byref(mode)):
                self._old_output_mode = int(mode.value)
                kernel32.SetConsoleMode(output_handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            os.system("")  # best-effort ANSI output support on classic consoles

    def _restore_windows_vt(self) -> None:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            if self._old_input_mode is not None:
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), self._old_input_mode)
            if self._old_output_mode is not None:
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), self._old_output_mode)
        except Exception:
            return


class TerminalKeyReader(AbstractContextManager["TerminalKeyReader"]):
    """Read terminal keys on one persistent background thread.

    ``Terminal.read_key`` is intentionally blocking.  Calling it through
    ``asyncio.to_thread`` once per key is fragile on Windows: when the console
    driver turns Ctrl+C into SIGINT, the awaiter can be interrupted while the
    worker thread remains stuck in ``msvcrt.getwch``.  Repeating that leaks one
    default-executor worker per Ctrl+C.  A single daemon reader thread bounds the
    damage to one blocked read and keeps unrelated ``asyncio.to_thread`` users
    from starving.
    """

    def __init__(
        self,
        terminal: Terminal,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        capture_sigint: bool | None = None,
    ) -> None:
        self.terminal = terminal
        self.loop = loop
        self.capture_sigint = terminal._windows if capture_sigint is None else capture_sigint
        self._queue: asyncio.Queue[str | Exception] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_sigint_handler = None
        self._sigint_installed = False

    def __enter__(self) -> "TerminalKeyReader":
        self.loop = self.loop or asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._install_sigint_handler()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="uv-agent-tui2-key-reader",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._restore_sigint_handler()
        # The OS-level terminal read may remain blocked until the next key.  The
        # thread is daemonized so shutdown is not held hostage by that read, but
        # join briefly for tests and EOF/error cases that can finish promptly.
        if self._thread is not None:
            self._thread.join(timeout=0.1)

    async def read_key(self) -> str:
        """Return the next key, re-raising reader failures in the event loop."""

        if self._queue is None:
            raise RuntimeError("TerminalKeyReader must be entered before reading")
        item = await self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                key = self.terminal.read_key()
            except KeyboardInterrupt:
                self._put_key("\x03")
                continue
            except Exception as exc:
                if not self._stop.is_set():
                    self._put_key(exc)
                return
            if not key:
                # In-memory streams can return EOF immediately.  Avoid a tight
                # loop if a test or redirected stdin reaches the end.
                sleep(0.01)
                continue
            self._put_key(key)

    def _put_key(self, key: str | Exception) -> None:
        loop = self.loop
        queue = self._queue
        if loop is None or queue is None or self._stop.is_set():
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, key)
        except RuntimeError:
            # The loop may already be closing during interpreter shutdown.
            return

    def _install_sigint_handler(self) -> None:
        if not self.capture_sigint:
            return
        try:
            import signal

            self._old_sigint_handler = signal.getsignal(signal.SIGINT)

            def _handle_sigint(signum, frame) -> None:  # noqa: ANN001 - signal handler API
                self._put_key("\x03")

            signal.signal(signal.SIGINT, _handle_sigint)
            self._sigint_installed = True
        except (OSError, RuntimeError, ValueError):
            # ``signal.signal`` only works in the main thread.  If tui2 is ever
            # embedded elsewhere, the reader thread still prevents executor
            # starvation and raw ETX input still works when the console permits.
            self._old_sigint_handler = None
            self._sigint_installed = False

    def _restore_sigint_handler(self) -> None:
        if not self._sigint_installed:
            return
        try:
            import signal

            signal.signal(signal.SIGINT, self._old_sigint_handler)
        except (OSError, RuntimeError, ValueError):
            return
        finally:
            self._sigint_installed = False
            self._old_sigint_handler = None
