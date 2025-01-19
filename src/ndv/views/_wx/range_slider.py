"""Adapted from https://gist.github.com/gabrieldp/e19611abead7f6617872d33866c568a3.

credit:
Gabriel Pasa
gabrieldp
"""

from __future__ import annotations

from typing import Any

import wx


def fraction_to_value(fraction: float, min_value: float, max_value: float) -> float:
    return (max_value - min_value) * fraction + min_value


def value_to_fraction(value: float, min_value: float, max_value: float) -> float:
    return float(value - min_value) / (max_value - min_value)


class SliderThumb:
    def __init__(self, parent: RangeSlider, value: int):
        self.parent = parent
        self.dragged = False
        self.mouse_over = False
        self.thumb_poly = ((0, 0), (0, 13), (5, 18), (10, 13), (10, 0))
        self.thumb_shadow_poly = ((0, 14), (4, 18), (6, 18), (10, 14))
        min_coords = [float("Inf"), float("Inf")]
        max_coords = [-float("Inf"), -float("Inf")]
        for pt in list(self.thumb_poly) + list(self.thumb_shadow_poly):
            for i_coord, coord in enumerate(pt):
                if coord > max_coords[i_coord]:
                    max_coords[i_coord] = coord
                if coord < min_coords[i_coord]:
                    min_coords[i_coord] = coord
        self.size = (max_coords[0] - min_coords[0], max_coords[1] - min_coords[1])

        self.value = value
        self.normal_color = wx.Colour((0, 120, 215))
        self.normal_shadow_color = wx.Colour((120, 180, 228))
        self.dragged_color = wx.Colour((204, 204, 204))
        self.dragged_shadow_color = wx.Colour((222, 222, 222))
        self.mouse_over_color = wx.Colour((23, 23, 23))
        self.mouse_over_shadow_color = wx.Colour((132, 132, 132))

    def GetPosition(self) -> tuple[int, int]:
        min_x = self.GetMin()
        max_x = self.GetMax()
        parent_size = self.parent.GetSize()
        min_value = self.parent.GetMin()
        max_value = self.parent.GetMax()
        fraction = value_to_fraction(self.value, min_value, max_value)
        low = int(fraction_to_value(fraction, min_x, max_x))
        high = int(parent_size[1] / 2 + 1)
        return low, high

    def SetPosition(self, pos: tuple[int, int]) -> None:
        pos_x = pos[0]
        # Limit movement by the position of the other thumb
        who_other, other_thumb = self.GetOtherThumb()
        other_pos = other_thumb.GetPosition()
        if who_other == "low":
            pos_x = int(
                max(other_pos[0] + other_thumb.size[0] / 2 + self.size[0] / 2, pos_x)
            )
        else:
            pos_x = int(
                min(other_pos[0] - other_thumb.size[0] / 2 - self.size[0] / 2, pos_x)
            )
        # Limit movement by slider boundaries
        min_x = self.GetMin()
        max_x = self.GetMax()
        pos_x = min(max(pos_x, min_x), max_x)

        fraction = value_to_fraction(pos_x, min_x, max_x)
        self.value = int(
            fraction_to_value(fraction, self.parent.GetMin(), self.parent.GetMax())
        )
        # Post event notifying that position changed
        self.PostEvent()

    def GetValue(self) -> int:
        return self.value

    def SetValue(self, value: int) -> None:
        self.value = value

    def PostEvent(self) -> None:
        event = wx.PyCommandEvent(wx.EVT_SLIDER.typeId, self.parent.GetId())
        event.SetEventObject(self.parent)
        wx.PostEvent(self.parent.GetEventHandler(), event)

    def GetMin(self) -> int:
        return self.parent.border_width + int(self.size[0] / 2)

    def GetMax(self) -> int:
        parent_w = int(self.parent.GetSize()[0])
        return parent_w - self.parent.border_width - int(self.size[0] / 2)

    def IsMouseOver(self, mouse_pos: wx.Point) -> bool:
        in_hitbox = True
        my_pos = self.GetPosition()
        for i_coord, mouse_coord in enumerate(mouse_pos):
            boundary_low = my_pos[i_coord] - self.size[i_coord] / 2
            boundary_high = my_pos[i_coord] + self.size[i_coord] / 2
            in_hitbox = in_hitbox and (boundary_low <= mouse_coord <= boundary_high)
        return in_hitbox

    def GetOtherThumb(self) -> tuple[str, SliderThumb]:
        if self.parent.thumbs["low"] != self:
            return "low", self.parent.thumbs["low"]
        else:
            return "high", self.parent.thumbs["high"]

    def OnPaint(self, dc: wx.BufferedPaintDC) -> None:
        if self.dragged or not self.parent.IsEnabled():
            thumb_color = self.dragged_color
            thumb_shadow_color = self.dragged_shadow_color
        elif self.mouse_over:
            thumb_color = self.mouse_over_color
            thumb_shadow_color = self.mouse_over_shadow_color
        else:
            thumb_color = self.normal_color
            thumb_shadow_color = self.normal_shadow_color
        my_pos = self.GetPosition()

        # Draw thumb shadow (or anti-aliasing effect)
        dc.SetBrush(wx.Brush(thumb_shadow_color, style=wx.BRUSHSTYLE_SOLID))
        dc.SetPen(wx.Pen(thumb_shadow_color, width=1, style=wx.PENSTYLE_SOLID))
        dc.DrawPolygon(
            points=self.thumb_shadow_poly,
            xoffset=int(my_pos[0] - self.size[0] / 2),
            yoffset=int(my_pos[1] - self.size[1] / 2),
        )
        # Draw thumb itself
        dc.SetBrush(wx.Brush(thumb_color, style=wx.BRUSHSTYLE_SOLID))
        dc.SetPen(wx.Pen(thumb_color, width=1, style=wx.PENSTYLE_SOLID))
        dc.DrawPolygon(
            points=self.thumb_poly,
            xoffset=int(my_pos[0] - self.size[0] / 2),
            yoffset=int(my_pos[1] - self.size[1] / 2),
        )


class RangeSlider(wx.Panel):
    def __init__(
        self,
        parent: wx.Window,
        id: int = wx.ID_ANY,
        lowValue: int | None = None,
        highValue: int | None = None,
        minValue: int = 0,
        maxValue: int = 100,
        pos: wx.Point = wx.DefaultPosition,
        size: wx.Size = wx.DefaultSize,
        style: int = wx.SL_HORIZONTAL,
        validator: wx.Validator = wx.DefaultValidator,
        name: str = "rangeSlider",
    ) -> None:
        if style != wx.SL_HORIZONTAL:
            raise NotImplementedError("Styles not implemented")
        if validator != wx.DefaultValidator:
            raise NotImplementedError("Validator not implemented")
        super().__init__(parent=parent, id=id, pos=pos, size=size, name=name)
        self.SetMinSize(size=(max(50, size[0]), max(26, size[1])))
        if minValue > maxValue:
            minValue, maxValue = maxValue, minValue
        self.min_value = minValue
        self.max_value = maxValue
        if lowValue is None:
            lowValue = self.min_value
        if highValue is None:
            highValue = self.max_value
        if lowValue > highValue:
            lowValue, highValue = highValue, lowValue
        lowValue = max(lowValue, self.min_value)
        highValue = min(highValue, self.max_value)
        self.track_height = 8
        self.border_width = 8

        self.thumbs = {
            "low": SliderThumb(parent=self, value=lowValue),
            "high": SliderThumb(parent=self, value=highValue),
        }
        self.thumb_width = self.thumbs["low"].size[0]

        # Aesthetic definitions
        self.slider_background_color = wx.Colour((231, 234, 234))
        self.slider_outline_color = wx.Colour((214, 214, 214))
        self.selected_range_color = wx.Colour((0, 120, 215))
        self.selected_range_outline_color = wx.Colour((0, 120, 215))

        # Bind events
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self.OnMouseLost)
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeave)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnResize)

    def Enable(self, enable: bool = True) -> None:
        super().Enable(enable)
        self.Refresh()

    def Disable(self) -> None:
        super().Disable()
        self.Refresh()

    def SetValueFromMousePosition(self, click_pos: wx.Point) -> None:
        for thumb in self.thumbs.values():
            if thumb.dragged:
                thumb.SetPosition(click_pos)

    def OnMouseDown(self, evt: wx.Event) -> None:
        if not self.IsEnabled():
            return
        click_pos = evt.GetPosition()
        for thumb in self.thumbs.values():
            if thumb.IsMouseOver(click_pos):
                thumb.dragged = True
                thumb.mouse_over = False
                break
        self.SetValueFromMousePosition(click_pos)
        self.CaptureMouse()
        self.Refresh()

    def OnMouseUp(self, evt: wx.Event) -> None:
        if not self.IsEnabled():
            return
        self.SetValueFromMousePosition(evt.GetPosition())
        for thumb in self.thumbs.values():
            thumb.dragged = False
        if self.HasCapture():
            self.ReleaseMouse()
        self.Refresh()

    def OnMouseLost(self, evt: wx.Event) -> None:
        for thumb in self.thumbs.values():
            thumb.dragged = False
            thumb.mouse_over = False
        self.Refresh()

    def OnMouseMotion(self, evt: wx.Event) -> None:
        if not self.IsEnabled():
            return
        refresh_needed = False
        mouse_pos = evt.GetPosition()
        if evt.Dragging() and evt.LeftIsDown():
            self.SetValueFromMousePosition(mouse_pos)
            refresh_needed = True
        else:
            for thumb in self.thumbs.values():
                old_mouse_over = thumb.mouse_over
                thumb.mouse_over = thumb.IsMouseOver(mouse_pos)
                if old_mouse_over != thumb.mouse_over:
                    refresh_needed = True
        if refresh_needed:
            self.Refresh()

    def OnMouseEnter(self, evt: wx.Event) -> None:
        if not self.IsEnabled():
            return
        mouse_pos = evt.GetPosition()
        for thumb in self.thumbs.values():
            if thumb.IsMouseOver(mouse_pos):
                thumb.mouse_over = True
                self.Refresh()
                break

    def OnMouseLeave(self, evt: wx.Event) -> None:
        if not self.IsEnabled():
            return
        for thumb in self.thumbs.values():
            thumb.mouse_over = False
        self.Refresh()

    def OnResize(self, evt: wx.Event) -> None:
        self.Refresh()

    def OnPaint(self, evt: wx.Event) -> None:
        w, h = self.GetSize()
        # BufferedPaintDC should reduce flickering
        dc = wx.BufferedPaintDC(self)
        background_brush = wx.Brush(self.GetBackgroundColour(), wx.SOLID)
        dc.SetBackground(background_brush)
        dc.Clear()
        # Draw slider
        dc.SetPen(wx.Pen(self.slider_outline_color, width=1, style=wx.PENSTYLE_SOLID))
        dc.SetBrush(wx.Brush(self.slider_background_color, style=wx.BRUSHSTYLE_SOLID))

        dc.DrawRectangle(
            int(self.border_width),
            int(h / 2 - self.track_height / 2),
            int(w - 2 * self.border_width),
            self.track_height,
        )
        # Draw selected range
        if self.IsEnabled():
            dc.SetPen(
                wx.Pen(
                    self.selected_range_outline_color, width=1, style=wx.PENSTYLE_SOLID
                )
            )
            dc.SetBrush(wx.Brush(self.selected_range_color, style=wx.BRUSHSTYLE_SOLID))
        else:
            dc.SetPen(
                wx.Pen(self.slider_outline_color, width=1, style=wx.PENSTYLE_SOLID)
            )
            dc.SetBrush(wx.Brush(self.slider_outline_color, style=wx.BRUSHSTYLE_SOLID))
        low_pos = self.thumbs["low"].GetPosition()[0]
        high_pos = self.thumbs["high"].GetPosition()[0]
        dc.DrawRectangle(
            int(low_pos),
            int(h / 2 - self.track_height / 4),
            int(high_pos - low_pos),
            int(self.track_height / 2),
        )
        # Draw thumbs
        for thumb in self.thumbs.values():
            thumb.OnPaint(dc)
        evt.Skip()

    def OnEraseBackground(self, evt: Any) -> None:
        # This should reduce flickering
        pass

    def GetValues(self) -> tuple[int, int]:
        return self.thumbs["low"].value, self.thumbs["high"].value

    def SetValue(self, lowValue: float, highValue: float) -> None:
        if lowValue > highValue:
            lowValue, highValue = highValue, lowValue
        lowValue = max(lowValue, self.min_value)
        highValue = min(highValue, self.max_value)
        self.thumbs["low"].SetValue(int(lowValue))
        self.thumbs["high"].SetValue(int(highValue))
        self.Refresh()

    def GetMax(self) -> int:
        return self.max_value

    def GetMin(self) -> int:
        return self.min_value

    def SetMax(self, maxValue: int) -> None:
        if maxValue < self.min_value:
            maxValue = self.min_value
        _, old_high = self.GetValues()
        if old_high > maxValue:
            self.thumbs["high"].SetValue(maxValue)
        self.max_value = maxValue
        self.Refresh()

    def SetMin(self, minValue: int) -> None:
        if minValue > self.max_value:
            minValue = self.max_value
        old_low, _ = self.GetValues()
        if old_low < minValue:
            self.thumbs["low"].SetValue(minValue)
        self.min_value = minValue
        self.Refresh()
