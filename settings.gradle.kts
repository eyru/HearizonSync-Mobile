// Google Maven 的 CDN 直连地址（maven.google.com 在本机不可达，此域名可达）
pluginManagement {
    repositories {
        maven { url = uri("https://dl.google.com/dl/android/maven2") }
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven { url = uri("https://dl.google.com/dl/android/maven2") }
        mavenCentral()
    }
}

rootProject.name = "HearizonSync"
include(":app")
