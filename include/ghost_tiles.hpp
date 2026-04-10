/*
 * ghost_tiles.hpp — Learned sparse attention patterns in C++17
 *
 * Part of the Lucineer fleet ecosystem.
 * Modern C++17 with RAII, std::vector, std::optional, ranges.
 *
 * See: https://github.com/Lucineer/ghost-tiles-cpp
 *
 * Integration:
 *   - cuda-ghost-tiles (Rust): Same algorithm, native performance
 *   - ghost-tiles-c (C): C FFI compatible layout
 *   - ghost-tiles-cuda: GPU kernel acceleration
 *   - cuda-attention: Fleet saliency scoring
 */

#ifndef GHOST_TILES_HPP
#define GHOST_TILES_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <optional>
#include <numeric>
#include <unordered_map>
#include <functional>

namespace lucineer {

/// Fuse two confidences via harmonic mean (same as cuda-confidence crate)
inline double fuse_confidence(double a, double b) {
    const double inv = 1.0 / std::max(a, 1e-10) + 1.0 / std::max(b, 1e-10);
    return inv >= 1e10 ? 0.0 : 1.0 / inv;
}

inline uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/// A single tile in the attention grid
struct GhostTile {
    uint16_t row = 0;
    uint16_t col = 0;
    bool active = true;
    double weight = 1.0;
    uint64_t last_used_ms = 0;
    uint64_t use_count = 0;
    double confidence = 1.0;

    double importance() const { return weight * confidence; }

    void use(double conf) {
        active = true;
        ++use_count;
        last_used_ms = now_ms();
        confidence = fuse_confidence(confidence, conf);
        weight = 1.0 - (1.0 - weight) * 0.9;
    }

    void decay(double rate) {
        uint64_t age_s = (now_ms() - last_used_ms) / 1000;
        double d = std::exp(-rate * age_s / 60.0);
        weight *= d;
        confidence *= (1.0 - rate * 0.1);
        if (weight < 0.01) active = false;
    }
};

/// A learned sparse attention pattern
class GhostPattern {
public:
    GhostPattern(std::string id, uint16_t seq_len, uint16_t tile_size, double sparsity_budget)
        : id_(std::move(id)), tile_size_(tile_size), sparsity_budget_(sparsity_budget)
    {
        uint16_t grid = (seq_len + tile_size - 1) / tile_size;
        grid_rows_ = grid;
        grid_cols_ = grid;
        tiles_.reserve(grid * grid);
        for (uint16_t r = 0; r < grid; ++r)
            for (uint16_t c = 0; c < grid; ++c)
                tiles_.push_back({r, c, true, 1.0, 0, 0, 1.0});
    }

    void use_tile(uint16_t row, uint16_t col, double confidence) {
        ++total_uses_;
        auto it = std::find_if(tiles_.begin(), tiles_.end(),
            [&](const GhostTile& t) { return t.row == row && t.col == col; });
        if (it != tiles_.end()) it->use(confidence);
    }

    void prune() {
        size_t max_active = static_cast<size_t>(
            tiles_.size() * (1.0 - sparsity_budget_));
        size_t active = std::count_if(tiles_.begin(), tiles_.end(),
            [](const GhostTile& t) { return t.active; });
        if (active <= max_active) return;
        std::partial_sort(tiles_.begin(), tiles_.begin() + max_active, tiles_.end(),
            [](const GhostTile& a, const GhostTile& b) { return a.importance() > b.importance(); });
        for (size_t i = max_active; i < tiles_.size(); ++i)
            tiles_[i].active = false;
    }

    void decay(double rate) { for (auto& t : tiles_) t.decay(rate); }

    void rebalance() {
        prune();
        decay(0.1);
        size_t max_active = static_cast<size_t>(tiles_.size() * (1.0 - sparsity_budget_));
        size_t active = active_count();
        if (active < max_active) {
            auto inactive = tiles_ | std::views::filter([](const GhostTile& t) { return !t.active; });
            std::vector<GhostTile> sorted(inactive.begin(), inactive.end());
            std::ranges::sort(sorted, [](const GhostTile& a, const GhostTile& b) { return a.confidence > b.confidence; });
            size_t slots = max_active - active;
            for (size_t i = 0; i < std::min(slots, sorted.size()); ++i) {
                auto it = std::find_if(tiles_.begin(), tiles_.end(),
                    [&](const GhostTile& t) { return t.row == sorted[i].row && t.col == sorted[i].col; });
                if (it != tiles_.end()) { it->active = true; it->weight = 0.5; }
            }
        }
    }

    size_t active_count() const {
        return std::count_if(tiles_.begin(), tiles_.end(),
            [](const GhostTile& t) { return t.active; });
    }

    double sparsity() const {
        return tiles_.empty() ? 0.0 : 1.0 - static_cast<double>(active_count()) / tiles_.size();
    }

    double compute_cost() const {
        return tiles_.empty() ? 1.0 : static_cast<double>(active_count()) / tiles_.size();
    }

    double efficiency() const {
        int active = 0, heavy = 0;
        for (const auto& t : tiles_) {
            if (t.active) { ++active; if (t.use_count > 5) ++heavy; }
        }
        return active > 0 ? static_cast<double>(heavy) / active : 0.0;
    }

    std::vector<float> attention_mask(uint16_t seq_len) const {
        std::vector<float> mask(seq_len * seq_len, 0.0f);
        for (const auto& t : tiles_) {
            if (!t.active) continue;
            for (int r = t.row * tile_size_; r < t.row * tile_size_ + tile_size_ && r < seq_len; ++r)
                for (int c = t.col * tile_size_; c < t.col * tile_size_ + tile_size_ && c < seq_len; ++c)
                    mask[static_cast<size_t>(r) * seq_len + c] = static_cast<float>(t.weight);
        }
        return mask;
    }

    const std::string& id() const { return id_; }
    const std::vector<GhostTile>& tiles() const { return tiles_; }

private:
    std::string id_;
    std::vector<GhostTile> tiles_;
    uint16_t grid_rows_ = 0;
    uint16_t grid_cols_ = 0;
    uint16_t tile_size_ = 8;
    double sparsity_budget_ = 0.5;
    uint64_t total_uses_ = 0;
};

/// Manages multiple ghost patterns
class GhostTileManager {
public:
    explicit GhostTileManager(double sparsity_budget) : sparsity_budget_(sparsity_budget) {}

    void add(GhostPattern pattern) {
        total_saved_ += static_cast<uint64_t>(pattern.compute_cost() * pattern.tiles().size());
        patterns_.emplace(pattern.id(), std::move(pattern));
    }

    const GhostPattern* best() const {
        if (patterns_.empty()) return nullptr;
        return &std::ranges::max_element(patterns_, {},
            [](const auto& kv) { return kv.second.efficiency(); })->second;
    }

    const GhostPattern* most_used() const {
        if (patterns_.empty()) return nullptr;
        return &std::ranges::max_element(patterns_, {},
            [](const auto& kv) { return kv.second.tiles().size() > 0 ? 0 : 0; })->second;
    }

    double avg_cost() const {
        if (patterns_.empty()) return 1.0;
        double sum = 0.0;
        for (const auto& [_, p] : patterns_) sum += p.compute_cost();
        return sum / patterns_.size();
    }

    double savings_pct() const {
        if (patterns_.empty()) return 0.0;
        uint64_t total = 0;
        for (const auto& [_, p] : patterns_) total += p.tiles().size();
        return total == 0 ? 0.0 : (1.0 - static_cast<double>(total_saved_) / total) * 100.0;
    }

private:
    std::unordered_map<std::string, GhostPattern> patterns_;
    double sparsity_budget_ = 0.5;
    uint64_t total_saved_ = 0;
};

} // namespace lucineer

#endif
