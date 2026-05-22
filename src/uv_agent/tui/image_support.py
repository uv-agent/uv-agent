from __future__ import annotations

from rich.cells import cell_len
from textual.css.query import NoMatches
from textual.geometry import Offset
from textual.widgets import Static, TextArea

from uv_agent.tui.panels import PendingImagePreviewPanel
from uv_agent.tui.state import PendingImage
from uv_agent.tui.widgets import TranscriptScroll


class ImageSupportMixin:
    def _refresh_pending_images(self) -> None:
        try:
            button = self.query_one("#pending-images-btn", Static)
        except NoMatches:
            return
        count = len(self._pending_images)
        if count:
            button.update(self._pending_image_count_label(count))
        else:
            button.update("")
        self._refresh_composer_overlay()

    def _refresh_pending_turns(self) -> None:
        try:
            button = self.query_one("#pending-turns-btn", Static)
        except NoMatches:
            return
        count = self._active_queue_length()
        if count:
            button.update(self._pending_turn_count_label(count))
        else:
            button.update("")
        self._refresh_composer_overlay()

    def _pending_image_count_label(self, count: int) -> str:
        if self.language == "zh":
            return f"{self._text('image_label')} {count}"
        image_word = "image" if count == 1 else self._text("images")
        return f"{count} {image_word}"

    def _pending_turn_count_label(self, count: int) -> str:
        if self.language == "zh":
            return f"{self._text('pending')} {count}"
        return f"{self._text('queued_title')} {count}"

    def _refresh_composer_overlay(self) -> None:
        try:
            turn_button = self.query_one("#pending-turns-btn", Static)
            image_button = self.query_one("#pending-images-btn", Static)
            bottom_button = self.query_one("#scroll-to-bottom-btn", Static)
            composer = self.query_one("#composer", TextArea)
            transcript = self.query_one("#transcript", TranscriptScroll)
        except NoMatches:
            return
        show_turns = self._active_queue_length() > 0
        show_images = bool(self._pending_images)
        near_bottom = self._transcript_is_near_bottom(transcript)
        if transcript.near_bottom != near_bottom:
            transcript.near_bottom = near_bottom
        show_bottom = not near_bottom
        turn_button.set_class(not show_turns, "hidden")
        image_button.set_class(not show_images, "hidden")
        bottom_button.set_class(not show_bottom, "hidden")
        if not (show_turns or show_images or show_bottom):
            turn_button.refresh(layout=True)
            image_button.refresh(layout=True)
            bottom_button.refresh(layout=True)
            return
        overlay_y = max(0, composer.region.y - 1)
        left_x = composer.region.x
        gap = 1
        current_x = left_x
        # New pending-send controls are intentionally grouped on the left edge
        # of the composer overlay; the existing scroll-to-bottom affordance
        # stays right-aligned so muscle memory is unaffected.
        for button, show in ((turn_button, show_turns), (image_button, show_images)):
            if not show:
                continue
            button.absolute_offset = Offset(current_x, overlay_y)
            current_x += self._overlay_button_width(button) + gap
        if show_bottom:
            right_width = self._overlay_button_width(bottom_button)
            right_x = max(left_x, composer.region.x + composer.region.width - right_width)
            bottom_button.absolute_offset = Offset(right_x, overlay_y)
        turn_button.refresh(layout=True)
        image_button.refresh(layout=True)
        bottom_button.refresh(layout=True)

    def _transcript_is_near_bottom(self, transcript: TranscriptScroll) -> bool:
        if transcript.max_scroll_y <= 0:
            return True
        return (transcript.max_scroll_y - transcript.scroll_y) <= transcript._BOTTOM_THRESHOLD

    def _overlay_button_width(self, button: Static) -> int:
        return max(1, cell_len(str(button.render())) + 4)

    def _open_pending_image_preview(self) -> None:
        if not self._pending_images:
            self._flash(self._text("no_pending_images"))
            return
        self.push_screen(PendingImagePreviewPanel(list(self._pending_images), len(self._pending_images) - 1))

    def _delete_pending_image(self, image: PendingImage) -> None:
        try:
            self._pending_images.remove(image)
        except ValueError:
            return
        self._refresh_pending_images()
        self._flash(f"{self._text('deleted_pending_image')}: {image.path.name}")
