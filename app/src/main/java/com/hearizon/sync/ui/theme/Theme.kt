package com.hearizon.sync.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Ink = Color(0xFF0A1020)
private val InkSoft = Color(0xFF131C2E)
private val DeepTeal = Color(0xFF0E7490)
private val Aqua = Color(0xFF22D3EE)
private val Paper = Color(0xFFF6F9FC)
private val Slate = Color(0xFF64748B)

private val DarkScheme = darkColorScheme(
    primary = Aqua,
    onPrimary = Ink,
    primaryContainer = Color(0xFF0E4A5C),
    onPrimaryContainer = Aqua,
    secondary = DeepTeal,
    onSecondary = Color.White,
    background = Ink,
    onBackground = Color(0xFFE6EDF7),
    surface = InkSoft,
    onSurface = Color(0xFFE6EDF7),
    surfaceVariant = Color(0xFF1E293B),
    onSurfaceVariant = Color(0xFF94A3B8),
    outline = Color(0xFF334155),
)

private val LightScheme = lightColorScheme(
    primary = DeepTeal,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFCFFAFE),
    onPrimaryContainer = Color(0xFF155E75),
    secondary = Aqua,
    onSecondary = Ink,
    background = Paper,
    onBackground = Ink,
    surface = Color.White,
    onSurface = Ink,
    surfaceVariant = Color(0xFFE7EEF6),
    onSurfaceVariant = Slate,
    outline = Color(0xFFCBD5E1),
)

@Composable
fun HearizonTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkScheme else LightScheme,
        content = content,
    )
}
