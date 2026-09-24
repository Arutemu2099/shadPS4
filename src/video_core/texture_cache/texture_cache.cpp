// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include <boost/container/small_vector.hpp>
#include <cstdlib>
#include <immintrin.h>
#include <thread>
#include <limits>
#include <xxhash.h>

#include "common/assert.h"
#include "common/path_util.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/fault_monitor.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           AmdGpu::Liverpool* liverpool_, BufferCache& buffer_cache_,
                           PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {
    // 4 MB: one zeroed counter per 1 MB page of the 40-bit guest space.
    page_image_counts = std::make_unique<std::atomic<u32>[]>(
        1ULL << (Traits::AddressSpaceBits - Traits::PageBits));

    const auto enabled = [](const char* name) {
        const auto* value = std::getenv(name);
        return value && value[0] == '1' && value[1] == '\0';
    };
    if (enabled("SHADPS4_PREPARED_IMAGES")) {
        prepared_images = std::make_unique<PreparedImages>();
        prepared_images->verify = enabled("SHADPS4_PREPARED_IMAGES_VERIFY");
    }

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    pressure_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_PRESSURE_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() {
    scheduler.DrainRecording();
}

void TextureCache::ProcessDownloadImages() {
    std::unique_lock lk{download_images_mutex};
    static const bool probe_enabled = [] {
        const auto* value = std::getenv("SHADPS4_IMAGE_DOWNLOAD_PROBE");
        return value && value[0] == '1';
    }();
    if (probe_enabled && !download_images.empty()) {
        // Infrequent batch-shape reporting only. No per-draw clocks or atomics.
        static u64 calls{}, copies{}, bytes{}, multi{}, fits{}, max_copies{}, max_bytes{};
        u64 batch_copies{}, batch_bytes{};
        for (const ImageId id : download_images) {
            const auto& image = slot_images[id];
            if (False(image.flags & ImageFlagBits::GpuModified)) {
                continue;
            }
            const u64 size = u64(image.info.pitch) * image.info.size.height *
                             image.info.size.depth * image.info.resources.layers *
                             (image.info.num_bits / 8);
            ++batch_copies;
            batch_bytes += (size + 15) & ~u64{15};
        }
        ++calls;
        copies += batch_copies;
        bytes += batch_bytes;
        multi += batch_copies > 1;
        fits += batch_copies > 1 &&
                batch_bytes <= buffer_cache.GetUtilityBuffer(MemoryUsage::Download).SizeBytes();
        max_copies = std::max(max_copies, batch_copies);
        max_bytes = std::max(max_bytes, batch_bytes);
        if ((calls & 1023) == 0) {
            const auto& sample = slot_images[*download_images.begin()];
            LOG_INFO(Render_Vulkan,
                     "ImageDownloadProbe: calls={} copies={} padded_bytes={} multi={} fits={} "
                     "max_copies={} max_bytes={} first_uid={} addr={:#x} size={}x{}x{} pitch={} "
                     "layers={} bits={} tiled={} block={} format={}",
                     calls, copies, bytes, multi, fits, max_copies, max_bytes, sample.image_uid,
                     sample.info.guest_address, sample.info.size.width, sample.info.size.height,
                     sample.info.size.depth, sample.info.pitch, sample.info.resources.layers,
                     sample.info.num_bits, u32(sample.info.props.is_tiled),
                     u32(sample.info.props.is_block), u32(sample.info.pixel_format));
        }
    }
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    const auto [download, offset] = download_buffer.Map(download_size);
    download_buffer.Commit();
    const vk::BufferImageCopy image_download = {
        .bufferOffset = offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    cmdbuf.copyImageToBuffer(image.GetImage(),
                             Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal),
                             download_buffer.Handle(), image_download);

    if (sync) {
        scheduler.Finish();
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image.info.guest_address),
                                                  download, download_size);
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image.info.guest_address, download, download_size] {
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr), download,
                                                          download_size);
            });
    }
}

void TextureCache::DebugDumpImages(const std::vector<VAddr>& addresses, u32 tag) {
    std::vector<ImageId> ids;
    {
        std::scoped_lock lock{mutex};
        for (const VAddr address : addresses) {
            if (const auto it = page_table.find(address >> Traits::PageBits); it != nullptr) {
                for (const ImageId id : *it) {
                    if (slot_images[id].info.guest_address == address &&
                        std::ranges::find(ids, id) == ids.end()) {
                        ids.push_back(id);
                    }
                }
            }
        }
    }
    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    for (const ImageId id : ids) {
        Image& image = slot_images[id];
        const u32 width = image.info.size.width;
        const u32 height = image.info.size.height;
        const u32 bytes_per_texel =
            image.info.pixel_format == vk::Format::eD32SfloatS8Uint ? 4 : image.info.num_bits / 8;
        if (width == 0 || height == 0 || bytes_per_texel == 0 || image.info.props.is_block) {
            continue;
        }
        const auto aspect = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                      : vk::ImageAspectFlagBits::eColor;
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                          fmt::format("imgdump_{}_{:x}_{}x{}_{}_{}.raw", tag,
                                      image.info.guest_address, width, height,
                                      vk::to_string(image.info.pixel_format), image.image_uid);
        std::FILE* file = std::fopen(path.string().c_str(), "wb");
        if (!file) {
            continue;
        }
        const u32 row_bytes = width * bytes_per_texel;
        const u32 rows_per_band = std::max(1u, static_cast<u32>((16_MB) / row_bytes));
        for (u32 y = 0; y < height; y += rows_per_band) {
            const u32 rows = std::min(rows_per_band, height - y);
            const u32 band_size = rows * row_bytes;
            const auto [mapped, offset] = download_buffer.Map(band_size);
            download_buffer.Commit();
            const vk::BufferImageCopy region = {
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {aspect, 0, 0, 1},
                .imageOffset = {0, static_cast<s32>(y), 0},
                .imageExtent = {width, rows, 1},
            };
            scheduler.EndRendering();
            image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                          {});
            {
                auto command_access = scheduler.AcquireCommandBuffer();
                command_access.Handle().copyImageToBuffer(
                    image.GetImage(), image.backing->state.layout, download_buffer.Handle(),
                    region);
            }
            scheduler.Finish();
            std::fwrite(mapped, 1, band_size, file);
        }
        std::fclose(file);
        LOG_WARNING(Render_Vulkan, "ImageDump: wrote {}", path.string());
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

/// SHADPS4_TEX_UNTRACK_OUTSIDE=1: page-watcher decrements that a guest write fault's
/// invalidation collects under the texture-cache mutex, applied after it is released.
/// Measured 2026-09-24: the CP blocked on this mutex ~5.5% of its wall time while fault
/// handlers held it across UpdatePageWatchers -> VirtualProtect (itself behind the address-space
/// mutex the CP also takes). Watcher counts commute; a reordered re-track only costs a refault.
static thread_local boost::container::small_vector<std::pair<VAddr, u64>, 8>* g_deferred_untracks =
    nullptr;

static bool UntrackOutsideEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_TEX_UNTRACK_OUTSIDE");
        return value && value[0] == '1';
    }();
    return enabled;
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    // Every guest write fault lands here and used to take the mutex FindImage takes ~30k times
    // a frame on the CP, even for pages no image touches. Reading a zero count is equivalent to
    // having taken the mutex just before a concurrent registration, which the locked path
    // already allows; a nonzero count takes the locked path unchanged.
    static const bool precheck = [] {
        const char* value = std::getenv("SHADPS4_TEX_INVALIDATE_PRECHECK");
        return value && value[0] == '1';
    }();
    if (precheck) {
        static std::atomic<u64> calls{0}, skips{0};
        const u64 first = pages_start >> Traits::PageBits;
        const u64 last = (pages_end - 1) >> Traits::PageBits;
        bool any = last >= (1ULL << (Traits::AddressSpaceBits - Traits::PageBits));
        for (u64 page = first; !any && page <= last; ++page) {
            any = page_image_counts[page].load(std::memory_order_acquire) != 0;
        }
        const u64 n = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        if (!any) {
            skips.fetch_add(1, std::memory_order_relaxed);
        }
        if ((n & 0x7ffff) == 0) {
            LOG_INFO(Render_Vulkan, "TexInvalidatePrecheck: calls={} skipped={} ({:.1f}%)", n,
                     skips.load(), 100.0 * skips.load() / n);
        }
        if (!any) {
            return;
        }
    }
    boost::container::small_vector<std::pair<VAddr, u64>, 8> deferred;
    // SHADPS4_TEX_GUEST_SPIN=1: fault handlers spin for this mutex instead of sleeping on it, so
    // the command processor, which takes it ~30k times a frame, never pays a kernel wake-up in
    // its unlock. Measured 2026-09-24 on 386F5167: CP time in this mutex's lock/unlock system
    // calls was ~6% of its wall time.
    static const bool guest_spin = [] {
        const char* value = std::getenv("SHADPS4_TEX_GUEST_SPIN");
        return value && value[0] == '1';
    }();
    std::unique_lock lock{mutex, std::defer_lock};
    if (guest_spin) {
        u32 spins = 0;
        while (!lock.try_lock()) {
            if (++spins < 64) {
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
    } else {
        lock.lock();
    }
    if (UntrackOutsideEnabled()) {
        g_deferred_untracks = &deferred;
    }
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
    g_deferred_untracks = nullptr;
    lock.unlock();
    for (const auto& [untrack_addr, untrack_size] : deferred) {
        tracker.UpdatePageWatchers<false>(untrack_addr, untrack_size);
    }
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

bool TextureCache::BufferImageAliasDiag() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_BUFFER_IMAGE_ALIAS_DIAG");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void TextureCache::DiagBufferWriteOverImages(VAddr address, size_t size, bool formatted) {
    static u64 handled = 0, missed = 0, writes = 0, report_at = 1;
    static std::unordered_set<u64> seen;
    std::scoped_lock lock{mutex};
    ++writes;
    ForEachImageInRegion(address, size, [&](ImageId, Image& image) {
        const bool covered = formatted && image.info.guest_address == address;
        ++(covered ? handled : missed);
        const u64 key = image.info.guest_address ^ (u64(covered) << 63) ^ (u64(formatted) << 62) ^
                        (u64(address - image.info.guest_address) << 40);
        if (seen.size() < 96 && seen.insert(key).second) {
            LOG_WARNING(Render_Vulkan,
                        "BufferWriteOverImage: {} buf={:#x}+{:#x} formatted={} image={:#x}+{:#x} "
                        "offset={:#x} {}x{}x{} {} tiled={} usage tex={} storage={} rt={} depth={}",
                        covered ? "handled" : "MISSED", address, size, formatted,
                        image.info.guest_address, image.info.guest_size,
                        s64(address) - s64(image.info.guest_address), image.info.size.width,
                        image.info.size.height, image.info.size.depth,
                        vk::to_string(image.info.pixel_format), u32(image.info.props.is_tiled),
                        u32(image.usage.texture), u32(image.usage.storage),
                        u32(image.usage.render_target), u32(image.usage.depth_target));
        }
    });
    if (writes >= report_at) {
        LOG_INFO(Render_Vulkan, "BufferWriteOverImage totals: written_binds={} handled={} missed={}",
                 writes, handled, missed);
        report_at *= 2;
    }
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

void TextureCache::CopyStencilAspect(ImageId src_id, ImageId dst_id) {
    auto& src = slot_images[src_id];
    auto& dst = slot_images[dst_id];
    const auto aspect_of = [](const Image& image) {
        return (image.aspect_mask & vk::ImageAspectFlagBits::eStencil)
                   ? vk::ImageAspectFlagBits::eStencil
                   : vk::ImageAspectFlagBits::eColor;
    };
    const vk::ImageCopy region{
        .srcSubresource = {aspect_of(src), 0, 0, 1},
        .dstSubresource = {aspect_of(dst), 0, 0, 1},
        .extent = {std::min(src.info.size.width, dst.info.size.width),
                   std::min(src.info.size.height, dst.info.size.height), 1},
    };
    scheduler.EndRendering();
    src.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    dst.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    auto command_access = scheduler.AcquireCommandBuffer();
    command_access.Handle().copyImage(src.GetImage(), src.backing->state.layout, dst.GetImage(),
                                      dst.backing->state.layout, region);
}

void TextureCache::SaveStencilShadow(ImageId depth_id) {
    const VAddr address = slot_images[depth_id].info.guest_address;
    const auto size = slot_images[depth_id].info.size;
    ImageId shadow_id{};
    if (const auto it = stencil_shadows.find(address); it != stencil_shadows.end() &&
                                                       slot_images[it->second].info.size == size) {
        shadow_id = it->second;
    } else {
        ImageInfo info = slot_images[depth_id].info;
        info.pixel_format = vk::Format::eR8Uint;
        info.props.is_depth = false;
        info.props.has_stencil = false;
        info.num_bits = 8;
        info.meta_info = {};
        shadow_id = slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
        slot_images[shadow_id].flags = ImageFlagBits::Empty;
        stencil_shadows[address] = shadow_id;
        LOG_WARNING(Render_Vulkan, "StencilShadow: created {}x{} shadow for {:#x}", size.width,
                    size.height, address);
    }
    CopyStencilAspect(depth_id, shadow_id);
}

void TextureCache::RestoreStencilShadow(ImageId depth_id) {
    const auto it = stencil_shadows.find(slot_images[depth_id].info.guest_address);
    if (it == stencil_shadows.end() ||
        !(slot_images[it->second].info.size == slot_images[depth_id].info.size)) {
        return;
    }
    CopyStencilAspect(it->second, depth_id);
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        const auto new_image_id =
            slot_images.insert(instance, scheduler, blit_helper, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;

        if (cache_image.info.num_samples == 1 && new_info.num_samples == 1) {
            // Perform depth<->color copy using the intermediate copy buffer.
            if (instance.IsMaintenance8Supported()) {
                new_image.CopyImage(cache_image);
            } else {
                const auto& copy_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
                new_image.CopyImageWithBuffer(cache_image, copy_buffer.Handle(), 0);
            }
        } else if (cache_image.info.num_samples == 1 && new_info.props.is_depth &&
                   new_info.num_samples > 1) {
            // Perform a rendering pass to transfer the channels of source as samples in dest.
            cache_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::AccessFlagBits2::eShaderRead, {});
            new_image.Transit(vk::ImageLayout::eDepthAttachmentOptimal,
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {});
            blit_helper.ReinterpretColorAsMsDepth(
                new_info.size.width, new_info.size.height, new_info.num_samples,
                cache_image.info.pixel_format, new_info.pixel_format, cache_image.GetImage(),
                new_image.GetImage());
        } else {
            LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
        }

        static const bool stencil_shadow = [] {
            const char* value = std::getenv("SHADPS4_STENCIL_SHADOW");
            return value && value[0] == '1';
        }();
        if (stencil_shadow) {
            const bool old_stencil = slot_images[cache_image_id].info.props.has_stencil;
            const bool new_stencil = slot_images[new_image_id].info.props.has_stencil;
            if (old_stencil && !new_stencil) {
                SaveStencilShadow(cache_image_id);
            } else if (!old_stencil && new_stencil) {
                RestoreStencilShadow(new_image_id);
            }
        }

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    merged_image.CopyMip(cache_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    new_image.CopyImage(src_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);

    TrackImage(new_image_id);
    new_image.flags &= ~ImageFlagBits::Dirty;
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    // FINDIMAGE SPLIT PROBE (tier-2 leaf): is the 3.18 ms/frame bt_findimage an obvious waste (like
    // FindImageFromRange) or mandatory cache work? Split lock-acquire / ForEachImageInRegion walk /
    // match+resolve, and count perfect-hit% + pages walked + new-image(miss)%. Strip before ship.
    static u64 fi_calls = 0, fi_lock_ns = 0, fi_walk_ns = 0, fi_rest_ns = 0, fi_pages = 0,
               fi_hit = 0, fi_new = 0, fi_cands = 0;
    const auto fi_t0 = Common::FaultMonitor::Now();
    std::scoped_lock lock{mutex};
    const auto fi_t1 = Common::FaultMonitor::Now();

    PreparedImage* memo = nullptr;
    PreparedImageKey key{};
    ImageId memo_id{};
    u64 memo_uid{};
    u64 lookup_generation{};
    const auto input_range = desc.view_info.range;
    if (prepared_images) {
        ++prepared_images->calls;
        lookup_generation = prepared_images->generation.load(std::memory_order_acquire);
        constexpr u64 address_limit = u64{1} << Traits::AddressSpaceBits;
        if (info.guest_address && info.guest_address < address_limit && info.guest_size &&
            info.guest_size <= address_limit - info.guest_address &&
            info.pixel_format != vk::Format::eUndefined) {
            key = {info.guest_address, info.guest_size, info.size, info.type,
                   info.pixel_format,  info.resources,  desc.type, exact_fmt};
            // Address/format select a bounded slot; full field equality resolves collisions.
            const u64 address_hash = (info.guest_address >> 8) ^ (info.guest_address >> 25);
            const u64 index =
                (address_hash ^ (u64(info.pixel_format) << 3)) & (PreparedImages::Capacity - 1);
            memo = &prepared_images->entries[index];
            if (memo->generation == lookup_generation && memo->key == key && memo->id &&
                slot_images.is_allocated(memo->id)) {
                const auto& cached = slot_images[memo->id];
                // SlotId has no generation. Undefined-format stencil placeholders have no UID
                // and are never eligible. No Image pointer or Vulkan backing/view is retained.
                if (memo->uid && cached.image_uid == memo->uid &&
                    True(cached.flags & ImageFlagBits::Registered) &&
                    False(cached.flags & ImageFlagBits::Picked)) {
                    memo_id = memo->id;
                    memo_uid = memo->uid;
                }
            }
        }
        if (memo_id) {
            ++prepared_images->hits;
        } else {
            ++prepared_images->fallbacks;
        }
    }
    const bool reuse = memo_id && !prepared_images->verify;
    ImageIds image_ids;
    if (!reuse) {
        // Verification runs this original lookup once, not a second mutating resolve pass.
        ForEachImageInRegion(
            info.guest_address, info.guest_size,
            [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });
    }
    const auto fi_t2 = Common::FaultMonitor::Now();
    ++fi_calls;
    fi_lock_ns += Common::FaultMonitor::DiffNs(fi_t0, fi_t1);
    fi_walk_ns += Common::FaultMonitor::DiffNs(fi_t1, fi_t2);
    if (!reuse) {
        fi_pages +=
            ((info.guest_address + info.guest_size - 1) >> 20) - (info.guest_address >> 20) + 1;
    }
    fi_cands += image_ids.size();

    ImageId image_id = reuse ? memo_id : ImageId{};

    // Check for a perfect match first. Last qualifying candidate wins; registration/removal
    // invalidates ALL memos so reuse preserves this ordering, including overlapping aliases.
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    const bool fi_perfect = static_cast<bool>(image_id);
    const ImageId perfect_image_id = image_id;

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        ++fi_new;
        image_id = slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    if (prepared_images) {
        if (memo_id && prepared_images->verify) {
            ASSERT_MSG(image_id == memo_id && image.image_uid == memo_uid &&
                           desc.view_info.range == input_range,
                       "PreparedImages mismatch: cached={} actual={} cached_uid={} actual_uid={}",
                       memo_id.index, image_id.index, memo_uid, image.image_uid);
            ++prepared_images->verified;
        }
        const u64 final_generation = prepared_images->generation.load(std::memory_order_acquire);
        // Creation and overlap resolution are intentionally not memoized. Even an initial
        // perfect match can be freed/recreated by the resource check; don't publish that result.
        // CMASK/FMASK/HTILE metadata mutations do not enter any perfect-match predicate.
        if (memo && !memo_id && perfect_image_id && image_id == perfect_image_id &&
            image.image_uid && True(image.flags & ImageFlagBits::Registered) &&
            !(image.info.resources < info.resources) && final_generation == lookup_generation) {
            *memo = {key, image_id, image.image_uid, final_generation};
        }
        if ((prepared_images->calls & 0x3ffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "PreparedImages: calls={} hit={} invalidation={} fallback={} verified={} "
                     "(cumulative; invalidation=structural changes)",
                     prepared_images->calls, prepared_images->hits, final_generation - 1,
                     prepared_images->fallbacks, prepared_images->verified);
        }
    }

    if (fi_perfect) {
        ++fi_hit;
    }
    fi_rest_ns += Common::FaultMonitor::SinceNs(fi_t2);
    if (fi_calls >= 500000) {
        LOG_INFO(Render,
                 "FiSplit: calls={} lock={:.3f} walk={:.3f} rest={:.3f}us/call "
                 "perfect={:.1f}% new={:.1f}% pages={:.2f} cands={:.2f}",
                 fi_calls, static_cast<double>(fi_lock_ns) / 1e3 / fi_calls,
                 static_cast<double>(fi_walk_ns) / 1e3 / fi_calls,
                 static_cast<double>(fi_rest_ns) / 1e3 / fi_calls,
                 100.0 * static_cast<double>(fi_hit) / fi_calls,
                 100.0 * static_cast<double>(fi_new) / fi_calls,
                 static_cast<double>(fi_pages) / fi_calls,
                 static_cast<double>(fi_cands) / fi_calls);
        fi_calls = fi_lock_ns = fi_walk_ns = fi_rest_ns = fi_pages = fi_hit = fi_new = fi_cands = 0;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    // Only images whose guest_address == address can match here. RegisterImage registers an image
    // on every page it spans, so any such candidate is present on the single page containing
    // `address`. Walking every page of the (often multi-MB) texel-buffer range therefore did N
    // all-missing page-table lookups to find candidates that can only ever live on the first page -
    // measured ~1.45 us/call, ~8.9 ms/frame in the forest, always missing. One lookup is sufficient
    // and exact.
    ImageIds image_ids;
    if (const auto it = page_table.find(address >> Traits::PageBits); it != nullptr) {
        for (const ImageId image_id : *it) {
            Image& image = slot_images[image_id];
            if (image.info.guest_address != address) {
                continue;
            }
            if (ensure_valid && !image.SafeToDownload()) {
                continue;
            }
            image_ids.push_back(image_id);
        }
    }
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        // Serialize the traversal's temporary Picked flags with FindImage, including
        // their cleanup. Also publish any new stencil mapping under that same lock.
        // UpdateImage above acquires/releases the mutex itself; do not cover it here.
        std::scoped_lock lock{mutex};
        ImageId stencil_id{};
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId image_id, Image& image) {
                                 if (image.info.guest_address == desc.info.stencil_addr) {
                                     stencil_id = image_id;
                                 }
                             });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id =
                slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        // Inserting the placeholder can relocate SlotVector's storage, invalidating
        // the depth reference acquired above. Reacquire through its stable slot ID.
        stencil_image.AssociateDepth(image_id, slot_images[image_id].image_uid);
    }

    return slot_images[image_id].FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    if (auto barrier = in_buffer->GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                             vk::PipelineStageFlagBits2::eTransfer)) {
        if (AsyncUploadRecord()) {
            scheduler.Record([barrier = *barrier](vk::CommandBuffer cmdbuf) {
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &barrier,
                });
            });
        } else {
            scheduler.AcquireCommandBuffer().Handle().pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &barrier.value(),
            });
        }
    }

    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer->Handle(), in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    image.Upload(image_copies, buffer, offset);
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     u32* native_index) {
    const u64 hash = XXH3_64bits(&sampler, sizeof(sampler));

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] = samplers.try_emplace(hash, instance, sampler, border_color_base);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    if (native_index) {
        *native_index =
            Vulkan::NativeBindings::Get()->SamplerIndex(it.value().native, it->second.Handle());
    }
    return it->second.Handle();
}

void TextureCache::InvalidatePreparedImages() {
    if (prepared_images) {
        // Publish after mapping changes; invalidation also covers a competing candidate, not
        // just the selected ID. Retain the original caller locking / CP ownership contracts.
        const u64 previous = prepared_images->generation.fetch_add(1, std::memory_order_release);
        if (previous == std::numeric_limits<u64>::max()) {
            UNREACHABLE_MSG("PreparedImages generation exhausted");
        }
    }
}

namespace {
/// SHADPS4_IMAGE_CHURN_LOG=1 (diagnostic): logs register/unregister of images whose guest
/// address has been registered 3+ times - the texture cache recreating a surface - capped at
/// 400 lines. CP thread and fault paths both register, so the map is guarded.
void LogImageChurn(const char* what, const VideoCore::Image& image) {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_IMAGE_CHURN_LOG");
        return value && value[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::mutex churn_mutex;
    static std::unordered_map<VAddr, u32> registrations;
    static u32 lines = 0;
    std::scoped_lock lock{churn_mutex};
    const auto& info = image.info;
    u32& count = registrations[info.guest_address];
    if (what[0] == 'R') {
        ++count;
    }
    if (count < 3 || lines >= 400) {
        return;
    }
    ++lines;
    LOG_WARNING(Render_Vulkan,
                "ImageChurn: {} #{} addr={:#x} size={:#x} {}x{}x{} {} samples={} type={} "
                "tile={} layers={} mips={} depth={} uid={}",
                what, count, info.guest_address, info.guest_size, info.size.width,
                info.size.height, info.size.depth, vk::to_string(info.pixel_format),
                info.num_samples, static_cast<u32>(info.type), static_cast<u32>(info.tile_mode),
                info.resources.layers, info.resources.levels, info.props.is_depth,
                image.image_uid);
}
} // namespace

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    LogImageChurn("Register", image);
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        page_table[page].push_back(image_id);
        page_image_counts[page].fetch_add(1, std::memory_order_release);
    });
    InvalidatePreparedImages();
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    LogImageChurn("Unregister", image);
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
        page_image_counts[page].fetch_sub(1, std::memory_order_release);
    });
    InvalidatePreparedImages();
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        if (g_deferred_untracks) {
            g_deferred_untracks->emplace_back(addr, size);
        } else {
            tracker.UpdatePageWatchers<false>(addr, size);
        }
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    if (g_deferred_untracks) {
        g_deferred_untracks->emplace_back(image_begin, size);
    } else {
        tracker.UpdatePageWatchers<false>(image_begin, size);
    }
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    if (g_deferred_untracks) {
        g_deferred_untracks->emplace_back(addr, size);
    } else {
        tracker.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        auto& image = slot_images[image_id];
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        if (download) {
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const size_t lru_id = samplers.at(hash).lru_id;
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };

    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::TouchImage(const Image& image) {
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");
    InvalidatePreparedImages();

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
        InvalidatePreparedImages();
    });
}

} // namespace VideoCore
