package com.hearizon.sync

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.hearizon.sync.ui.AppScreen
import com.hearizon.sync.ui.theme.HearizonTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            HearizonTheme {
                AppScreen()
            }
        }
    }
}
