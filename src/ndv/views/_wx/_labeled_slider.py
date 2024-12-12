import wx


class WxLabeledSlider(wx.Panel):
    """A simple labeled slider widget for wxPython."""

    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent)

        self.label = wx.StaticText(self)
        self.slider = wx.Slider(self, style=wx.HORIZONTAL)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def setRange(self, min_val: int, max_val: int) -> None:
        self.slider.SetMin(min_val)
        self.slider.SetMax(max_val)

    def setValue(self, value: int) -> None:
        self.slider.SetValue(value)

    def value(self) -> int:
        return self.slider.GetValue()  # type: ignore [no-any-return]

    def setSingleStep(self, step: int) -> None:
        self.slider.SetLineSize(step)
