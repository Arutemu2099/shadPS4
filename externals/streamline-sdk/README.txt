NVIDIA Streamline SDK 2.14.16 public headers (MIT, see LICENSE.txt), copied from
https://github.com/NVIDIA-RTX/Streamline/releases/tag/v2.14.16 (streamline-sdk-v2.14.16.zip,
SHA-256 32041e51db0cab709a271b72a352281b690c2abbd669a827809401a5dd23956b) on 2026-10-17.
Only these headers are compiled in (SHADPS4_DLSS, src/video_core/renderer_vulkan/vk_dlss.cpp); no NVIDIA
binary is linked or shipped. At run time the player's own signed NVIDIA DLLs are loaded from dlss\ next to
shadPS4.exe (sl.interposer.dll, sl.common.dll, sl.dlss.dll, nvngx_dlss.dll); without them the game's own
anti-aliasing runs.
