// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;

// hdr: 0 = SDR frame on an SDR screen, 1 = HDR screen (passed through), 2 = the game's HDR10 (PQ)
// frame on an SDR screen (1.1.1, GitHub #13: decoded to SDR - the screenshot path's math).
layout (push_constant) uniform settings {
    float gamma;
    uint hdr;
} pp;

const float cutoff = 0.0031308, a = 1.055, b = 0.055, d = 12.92;
vec3 gamma(vec3 rgb) {
    return mix(
        a * pow(rgb, vec3(1.0 / (2.4 + 1.0 - pp.gamma))) - b,
        d * rgb / pp.gamma,
        lessThan(rgb, vec3(cutoff))
    );
}

// ST.2084 inverse EOTF: PQ code value -> nits.
vec3 pq_to_nits(vec3 e) {
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    vec3 vp = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
    vec3 num = max(vp - c1, vec3(0.0));
    vec3 den = max(c2 - c3 * vp, vec3(1e-6));
    return 10000.0 * pow(num / den, vec3(1.0 / m1));
}

vec3 pq_to_sdr_linear(vec3 e) {
    vec3 n2020 = pq_to_nits(e);
    // BT.2020 -> BT.709 primaries (rows), as in vk_presenter.cpp's screenshot conversion.
    vec3 n709 = vec3(dot(vec3(1.6605, -0.5876, -0.0728), n2020),
                     dot(vec3(-0.1246, 1.1329, -0.0083), n2020),
                     dot(vec3(-0.0182, -0.1006, 1.1187), n2020));
    // 100 nits = SDR white, highlights rolled off (2x / (1 + x)).
    vec3 x = max(n709, vec3(0.0)) / 100.0;
    return clamp(2.0 * x / (1.0 + x), 0.0, 1.0);
}

void main() {
    vec4 color_linear = texture(texSampler, uv);
    if (pp.hdr == 1u) {
        color = color_linear;
    } else if (pp.hdr == 2u) {
        color = vec4(gamma(pq_to_sdr_linear(color_linear.rgb)), color_linear.a);
    } else {
        color = vec4(gamma(color_linear.rgb), color_linear.a);
    }
}
