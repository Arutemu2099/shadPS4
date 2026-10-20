// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <fmt/format.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/update_notice.h"
#include "core/devtools/layer.h"

namespace Common {

namespace {
/// "second-son-v1.0.2" -> {1, 0, 2}; missing parts are 0.
std::array<int, 3> ParseTag(std::string_view tag) {
    std::array<int, 3> v{};
    const size_t p = tag.rfind('v');
    if (p == std::string_view::npos) {
        return v;
    }
    size_t i = p + 1;
    for (int& part : v) {
        int n = 0;
        bool any = false;
        while (i < tag.size() && tag[i] >= '0' && tag[i] <= '9') {
            n = n * 10 + (tag[i++] - '0');
            any = true;
        }
        if (!any) {
            break;
        }
        part = n;
        if (i >= tag.size() || tag[i] != '.') {
            break;
        }
        ++i;
    }
    return v;
}
} // namespace

void StartUpdateNotice() {
    const char* env = std::getenv("SHADPS4_UPDATE_CHECK");
    if (env && env[0] == '0') {
        return;
    }
    std::thread([] {
        SetCurrentThreadName("UpdateCheck");
        httplib::Client cli("https://api.github.com");
        cli.set_connection_timeout(5);
        cli.set_read_timeout(5);
        const httplib::Headers headers = {{"User-Agent", "shadPS4-second-son"},
                                          {"Accept", "application/vnd.github+json"}};
        const auto res = cli.Get("/repos/axylpup/shadPS4/releases/latest", headers);
        if (!res || res->status != 200) {
            LOG_INFO(Common, "UpdateCheck: no answer ({})",
                     res ? std::to_string(res->status) : httplib::to_string(res.error()));
            return;
        }
        const auto json = nlohmann::json::parse(res->body, nullptr, false);
        if (json.is_discarded() || !json.contains("tag_name") || !json["tag_name"].is_string()) {
            LOG_INFO(Common, "UpdateCheck: unexpected answer");
            return;
        }
        const std::string latest = json["tag_name"].get<std::string>();
        // SHADPS4_UPDATE_TEST_TAG=second-son-v1.0.0 (test): this build pretends to be that release.
        const char* test_tag = std::getenv("SHADPS4_UPDATE_TEST_TAG");
        const std::string_view mine = test_tag ? test_tag : ReleaseTag;
        if (ParseTag(latest) <= ParseTag(mine)) {
            LOG_INFO(Common, "UpdateCheck: {} is the latest release ({})", mine, latest);
            return;
        }
        LOG_INFO(Common, "UpdateCheck: {} is out (this build: {})", latest, mine);
        // After the boot screens; a notice is drawn only once the overlay exists.
        std::this_thread::sleep_for(std::chrono::seconds(45));
        const std::string_view name = std::string_view(latest).substr(latest.rfind('v'));
        Overlay::ShowNotice(
            fmt::format("Update available: {} - github.com/axylpup/shadPS4/releases", name).c_str(),
            10.0);
    }).detach();
}

} // namespace Common
