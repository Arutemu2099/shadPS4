// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Common {

/// The fork's release tag this build belongs to; the release checklist bumps it (ROADMAP).
inline constexpr const char* ReleaseTag = "second-son-v1.1";

/// 1.1 (LATER-PLAN 17): one background request to GitHub's latest release of axylpup/shadPS4 at
/// start; if it is newer than ReleaseTag, a 10 s notice after the boot screens. Nothing is
/// downloaded. SHADPS4_UPDATE_CHECK=0 turns it off.
void StartUpdateNotice();

} // namespace Common
