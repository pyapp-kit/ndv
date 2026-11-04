The @command
decorator and _collect_anywidget_commands / _register_anywidget_commands are defined in
anywidget.experimental (alongside widget, dataclass, etc.), which strongly implies they're meant to work
together. But:

1. _register_anywidget_commands(widget) is only called in AnyWidget.__init__ — the class-based path
2. ReprMimeBundle (the descriptor path) never calls it
3. So @command methods on a @widget-decorated class are silently dead code
4. model.invoke() from JS fails silently (or errors) because no command handler is registered on the comm

The expectation that @command works with @widget is very reasonable — they're in the same module and the
@dataclass decorator docs even show them together. The gap is that the descriptor path doesn't wire up
command registration.


---
Title: @command decorator doesn't work with MimeBundleDescriptor / @widget decorator

Summary: Methods decorated with @anywidget.experimental.command are never registered when using the
MimeBundleDescriptor (i.e., @widget or @dataclass from anywidget.experimental). Commands only work with the
class-based AnyWidget approach.

Root cause: _register_anywidget_commands() and _collect_anywidget_commands() are only called in
AnyWidget.__init__(). The ReprMimeBundle initialization path never invokes them, so the comm's on_msg
handler only processes method: "update" messages, not kind: "anywidget-command" messages.

Expected behavior: @command methods on a @widget-decorated class should work identically to @command methods
on an AnyWidget subclass. model.invoke("my_command", data) from JS should reach the Python method.

Suggested fix: ReprMimeBundle.__init__ (or sync_object_with_view) should call
_collect_anywidget_commands(type(obj)) and register a command handler on the comm, similar to what
AnyWidget.__init__ does.

Workaround: Use a regular synced field (e.g., _js_event: dict) that JS writes to via model.set() +
model.save_changes(), and observe changes on the Python side via psygnal to dispatch events.

Affected version: 0.9.21 and current main branch.
