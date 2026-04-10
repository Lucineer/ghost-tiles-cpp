/*
 * ghost-tiles.h — Learned sparse attention patterns in pure C
 *
 * Zero dependencies. C11. ARM64/x86-64/WASM portable.
 * Part of the Lucineer fleet ecosystem.
 * See: https://github.com/Lucineer/cuda-ghost-tiles
 */

#ifndef GHOST_TILES_H
#define GHOST_TILES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GT_MAX_TILES     1024
#define GT_MAX_PATTERNS  32
#define GT_MAX_ID_LEN    64

#define GT_OK             0
#define GT_ERR_BOUNDS     1
#define GT_ERR_NULL       2
#define GT_ERR_FULL       3

typedef struct {
    uint16_t row;
    uint16_t col;
    bool     active;
    double   weight;
    uint64_t last_used_ms;
    uint64_t use_count;
    double   confidence;
    double   importance;
} gt_tile;

typedef struct {
    char     id[GT_MAX_ID_LEN];
    gt_tile *tiles;
    uint16_t num_tiles;
    uint16_t grid_rows;
    uint16_t grid_cols;
    uint16_t tile_size;
    double   sparsity_budget;
    uint64_t total_uses;
    uint64_t created_ms;
} gt_pattern;

typedef struct {
    gt_pattern patterns[GT_MAX_PATTERNS];
    uint16_t   num_patterns;
    double     sparsity_budget;
    uint64_t   total_saved;
} gt_manager;

int gt_tile_use(gt_tile *t, double confidence, uint64_t now_ms);
int gt_tile_decay(gt_tile *t, double rate, uint64_t now_ms);
int gt_pattern_init(gt_pattern *p, const char *id,
                    uint16_t seq_len, uint16_t tile_size,
                    double sparsity_budget);
void gt_pattern_free(gt_pattern *p);
int gt_pattern_use(gt_pattern *p, uint16_t row, uint16_t col,
                   double confidence, uint64_t now_ms);
int gt_pattern_prune(gt_pattern *p);
int gt_pattern_decay(gt_pattern *p, double rate, uint64_t now_ms);
int gt_pattern_rebalance(gt_pattern *p, uint64_t now_ms);
int gt_pattern_active_count(const gt_pattern *p);
double gt_pattern_sparsity(const gt_pattern *p);
double gt_pattern_compute_cost(const gt_pattern *p);
double gt_pattern_efficiency(const gt_pattern *p);
int gt_pattern_attention_mask(const gt_pattern *p, float *mask, uint16_t seq_len);
int gt_pattern_merge(const gt_pattern *a, const gt_pattern *b,
                     gt_pattern *out, const char *new_id);
int gt_manager_init(gt_manager *m, double sparsity_budget);
int gt_manager_add(gt_manager *m, gt_pattern *p);
const gt_pattern *gt_manager_best(const gt_manager *m);
const gt_pattern *gt_manager_get(const gt_manager *m, const char *id);
double gt_manager_avg_cost(const gt_manager *m);
double gt_manager_savings_pct(const gt_manager *m);

#ifdef __cplusplus
}
#endif

#endif
