# adaptivecad_playground/curved_theme.py
"""
Theme presets for CurvedUI (no straight lines). Add more as needed.
"""
from dataclasses import dataclass

@dataclass
class CurvedThemePreset:
    name: str
    bg1: str
    bg2: str
    edge: str
    tick: str
    text: str
    btn_fill: str
    btn_fill_hi: str

JARVIS = CurvedThemePreset(
    name="jarvis",
    bg1="#081C28",
    bg2="#0C2A3A",
    edge="#00D1FF",
    tick="#00FFC2",
    text="#E6FCFF",
    btn_fill="#0F3347",
    btn_fill_hi="#10485F",
)

NEON = CurvedThemePreset(
    name="neon",
    bg1="#0A0814",
    bg2="#14122A",
    edge="#8A5CFF",
    tick="#00FFD1",
    text="#F0EEFF",
    btn_fill="#1E1840",
    btn_fill_hi="#2A1F5C",
)
