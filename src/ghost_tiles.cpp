/*
 * ghost-tiles.c — Learned sparse attention patterns in pure C
 * Zero dependencies. C11. ARM64/x86-64/WASM portable.
 */

#include "ghost-tiles.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static uint64_t gt_now_ms(void) {
    return (uint64_t)time(NULL) * 1000;
}

static double gt_fuse(double a, double b) {
    double a1 = a > 1e-10 ? a : 1e-10;
    double b1 = b > 1e-10 ? b : 1e-10;
    double inv = 1.0 / a1 + 1.0 / b1;
    if (inv >= 1e10) return 0.0;
    return 1.0 / inv;
}

static int tile_idx(const gt_pattern *p, uint16_t row, uint16_t col) {
    if (row >= p->grid_rows || col >= p->grid_cols) return -1;
    return row * p->grid_cols + col;
}

static int cmp_desc(const void *a, const void *b) {
    double ia = ((const gt_tile *)a)->importance;
    double ib = ((const gt_tile *)b)->importance;
    if (ib > ia) return 1;
    if (ib < ia) return -1;
    return 0;
}

/* ── Tile ── */
int gt_tile_use(gt_tile *t, double confidence, uint64_t now_ms) {
    if (!t) return GT_ERR_NULL;
    t->active = true;
    t->use_count++;
    t->last_used_ms = now_ms;
    t->confidence = gt_fuse(t->confidence, confidence);
    t->weight = 1.0 - (1.0 - t->weight) * 0.9;
    t->importance = t->weight * t->confidence;
    return GT_OK;
}

int gt_tile_decay(gt_tile *t, double rate, uint64_t now_ms) {
    if (!t) return GT_ERR_NULL;
    uint64_t age_s = (now_ms - t->last_used_ms) / 1000;
    double decay = exp(-rate * age_s / 60.0);
    t->weight *= decay;
    t->confidence *= (1.0 - rate * 0.1);
    t->importance = t->weight * t->confidence;
    if (t->weight < 0.01) t->active = false;
    return GT_OK;
}

/* ── Pattern ── */
int gt_pattern_init(gt_pattern *p, const char *id,
                    uint16_t seq_len, uint16_t tile_size,
                    double sparsity_budget) {
    if (!p || !id) return GT_ERR_NULL;
    uint16_t grid = (seq_len + tile_size - 1) / tile_size;
    p->tiles = (gt_tile *)calloc((size_t)grid * grid, sizeof(gt_tile));
    if (!p->tiles) return GT_ERR_FULL;
    strncpy(p->id, id, GT_MAX_ID_LEN - 1);
    p->id[GT_MAX_ID_LEN - 1] = '\0';
    p->grid_rows = grid;
    p->grid_cols = grid;
    p->num_tiles = grid * grid;
    p->tile_size = tile_size;
    p->sparsity_budget = sparsity_budget;
    p->total_uses = 0;
    p->created_ms = gt_now_ms();
    for (uint16_t r = 0; r < grid; r++) {
        for (uint16_t c = 0; c < grid; c++) {
            int i = tile_idx(p, r, c);
            p->tiles[i].row = r;
            p->tiles[i].col = c;
            p->tiles[i].active = true;
            p->tiles[i].weight = 1.0;
            p->tiles[i].confidence = 1.0;
            p->tiles[i].importance = 0.5;
        }
    }
    return GT_OK;
}

void gt_pattern_free(gt_pattern *p) {
    if (p && p->tiles) { free(p->tiles); p->tiles = NULL; }
}

int gt_pattern_use(gt_pattern *p, uint16_t row, uint16_t col,
                   double confidence, uint64_t now_ms) {
    if (!p) return GT_ERR_NULL;
    int i = tile_idx(p, row, col);
    if (i < 0) return GT_ERR_BOUNDS;
    p->total_uses++;
    return gt_tile_use(&p->tiles[i], confidence, now_ms);
}

int gt_pattern_prune(gt_pattern *p) {
    if (!p) return GT_ERR_NULL;
    int max_active = (int)((double)p->num_tiles * (1.0 - p->sparsity_budget));
    int active_count = 0;
    for (int i = 0; i < p->num_tiles; i++)
        if (p->tiles[i].active) active_count++;
    if (active_count <= max_active) return GT_OK;
    gt_tile *sorted = (gt_tile *)malloc(p->num_tiles * sizeof(gt_tile));
    memcpy(sorted, p->tiles, p->num_tiles * sizeof(gt_tile));
    qsort(sorted, p->num_tiles, sizeof(gt_tile), cmp_desc);
    for (int i = max_active; i < p->num_tiles; i++)
        sorted[i].active = false;
    for (int i = 0; i < p->num_tiles; i++) {
        int oi = tile_idx(p, sorted[i].row, sorted[i].col);
        if (oi >= 0) p->tiles[oi].active = sorted[i].active;
    }
    free(sorted);
    return GT_OK;
}

int gt_pattern_decay(gt_pattern *p, double rate, uint64_t now_ms) {
    if (!p) return GT_ERR_NULL;
    for (int i = 0; i < p->num_tiles; i++)
        gt_tile_decay(&p->tiles[i], rate, now_ms);
    return GT_OK;
}

int gt_pattern_rebalance(gt_pattern *p, uint64_t now_ms) {
    if (!p) return GT_ERR_NULL;
    gt_pattern_prune(p);
    gt_pattern_decay(p, 0.1, now_ms);
    int max_active = (int)((double)p->num_tiles * (1.0 - p->sparsity_budget));
    int active_count = gt_pattern_active_count(p);
    if (active_count < max_active) {
        int slots = max_active - active_count;
        for (int s = 0; s < slots; s++) {
            int best = -1; double bc = 0.0;
            for (int i = 0; i < p->num_tiles; i++) {
                if (!p->tiles[i].active && p->tiles[i].confidence > bc) {
                    bc = p->tiles[i].confidence; best = i;
                }
            }
            if (best >= 0) { p->tiles[best].active = true; p->tiles[best].weight = 0.5; }
        }
    }
    return GT_OK;
}

int gt_pattern_active_count(const gt_pattern *p) {
    if (!p) return 0;
    int n = 0;
    for (int i = 0; i < p->num_tiles; i++) if (p->tiles[i].active) n++;
    return n;
}

double gt_pattern_sparsity(const gt_pattern *p) {
    if (!p || p->num_tiles == 0) return 0.0;
    return 1.0 - (double)gt_pattern_active_count(p) / (double)p->num_tiles;
}

double gt_pattern_compute_cost(const gt_pattern *p) {
    if (!p || p->num_tiles == 0) return 1.0;
    return (double)gt_pattern_active_count(p) / (double)p->num_tiles;
}

double gt_pattern_efficiency(const gt_pattern *p) {
    if (!p) return 0.0;
    int active = 0, heavy = 0;
    for (int i = 0; i < p->num_tiles; i++) {
        if (p->tiles[i].active) { active++; if (p->tiles[i].use_count > 5) heavy++; }
    }
    return active > 0 ? (double)heavy / (double)active : 0.0;
}

int gt_pattern_attention_mask(const gt_pattern *p, float *mask, uint16_t seq_len) {
    if (!p || !mask) return GT_ERR_NULL;
    memset(mask, 0, sizeof(float) * (size_t)seq_len * seq_len);
    for (int i = 0; i < p->num_tiles; i++) {
        if (!p->tiles[i].active) continue;
        uint16_t r0 = p->tiles[i].row * p->tile_size;
        uint16_t c0 = p->tiles[i].col * p->tile_size;
        for (uint16_t r = r0; r < r0 + p->tile_size && r < seq_len; r++)
            for (uint16_t c = c0; c < c0 + p->tile_size && c < seq_len; c++)
                mask[(size_t)r * seq_len + c] = (float)p->tiles[i].weight;
    }
    return GT_OK;
}

int gt_pattern_merge(const gt_pattern *a, const gt_pattern *b,
                     gt_pattern *out, const char *new_id) {
    if (!a || !b || !out) return GT_ERR_NULL;
    int min_t = a->num_tiles < b->num_tiles ? a->num_tiles : b->num_tiles;
    gt_pattern_init(out, new_id, a->grid_rows * a->tile_size, a->tile_size,
                    (a->sparsity_budget + b->sparsity_budget) / 2.0);
    for (int i = 0; i < min_t && i < out->num_tiles; i++) {
        out->tiles[i].weight = (a->tiles[i].weight + b->tiles[i].weight) / 2.0;
        out->tiles[i].confidence = gt_fuse(a->tiles[i].confidence, b->tiles[i].confidence);
        out->tiles[i].use_count = a->tiles[i].use_count > b->tiles[i].use_count
                                   ? a->tiles[i].use_count : b->tiles[i].use_count;
        out->tiles[i].importance = out->tiles[i].weight * out->tiles[i].confidence;
        out->tiles[i].active = a->tiles[i].active || b->tiles[i].active;
    }
    gt_pattern_prune(out);
    return GT_OK;
}

/* ── Manager ── */
int gt_manager_init(gt_manager *m, double sparsity_budget) {
    if (!m) return GT_ERR_NULL;
    memset(m, 0, sizeof(gt_manager));
    m->sparsity_budget = sparsity_budget;
    return GT_OK;
}

int gt_manager_add(gt_manager *m, gt_pattern *p) {
    if (!m || !p) return GT_ERR_NULL;
    if (m->num_patterns >= GT_MAX_PATTERNS) return GT_ERR_FULL;
    m->patterns[m->num_patterns] = *p;
    m->total_saved += (uint64_t)((double)p->num_tiles * gt_pattern_compute_cost(p));
    m->num_patterns++;
    return GT_OK;
}

const gt_pattern *gt_manager_best(const gt_manager *m) {
    if (!m || m->num_patterns == 0) return NULL;
    int best = 0; double be = -1.0;
    for (int i = 0; i < m->num_patterns; i++) {
        double e = gt_pattern_efficiency(&m->patterns[i]);
        if (e > be) { be = e; best = i; }
    }
    return &m->patterns[best];
}

const gt_pattern *gt_manager_get(const gt_manager *m, const char *id) {
    if (!m || !id) return NULL;
    for (int i = 0; i < m->num_patterns; i++)
        if (strcmp(m->patterns[i].id, id) == 0) return &m->patterns[i];
    return NULL;
}

double gt_manager_avg_cost(const gt_manager *m) {
    if (!m || m->num_patterns == 0) return 1.0;
    double t = 0.0;
    for (int i = 0; i < m->num_patterns; i++)
        t += gt_pattern_compute_cost(&m->patterns[i]);
    return t / (double)m->num_patterns;
}

double gt_manager_savings_pct(const gt_manager *m) {
    if (!m || m->num_patterns == 0) return 0.0;
    uint64_t total = 0;
    for (int i = 0; i < m->num_patterns; i++) total += m->patterns[i].num_tiles;
    if (total == 0) return 0.0;
    return (1.0 - (double)m->total_saved / (double)total) * 100.0;
}
