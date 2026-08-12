from flimkit.plugins import tool

FLIMKIT_PLUGIN_API = 1

__version__ = '0.1.0'


@tool(id='anisotropy', label='Time-Resolved Anisotropy...', menu='Tools', order=20)
def open_anisotropy(app):
    from .tool import show_anisotropy_tool
    show_anisotropy_tool(app.root)
