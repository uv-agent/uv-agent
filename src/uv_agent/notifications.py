from __future__ import annotations

import os
import subprocess


def send_desktop_completion_notification(title: str, message: str) -> bool:
    """Best-effort desktop notification for turn completion."""
    if os.name != "nt":
        return False

    script = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Information
$notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
$notify.BalloonTipTitle = $args[0]
$notify.BalloonTipText = $args[1]
$notify.Visible = $true
$notify.ShowBalloonTip(5000)
Start-Sleep -Milliseconds 5500
$notify.Dispose()
"""
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
                title,
                message,
            ],
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return True
