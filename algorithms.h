#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

// --- 1. Data Structure ---
struct Point {
    double x, y;
};

// --- 2. The LTTB Logic (Largest Triangle Three Buckets) ---
inline std::vector<Point> LTTB(const std::vector<Point>& data, int threshold) {
    if (threshold >= data.size() || threshold <= 2) return data;

    std::vector<Point> sampled;
    sampled.reserve(threshold);

    // Always include the first point
    sampled.push_back(data[0]);

    double bin_size = (double)(data.size() - 2) / (threshold - 2);

    int a = 0; // Index of the previously selected point
    int next_a = 0;

    for (int i = 0; i < threshold - 2; ++i) {
        // Calculate the average of the *next* bucket to act as the third vertex
        double avg_x = 0, avg_y = 0;
        int avg_range_start = (int)floor((i + 1) * bin_size) + 1;
        int avg_range_end = (int)floor((i + 2) * bin_size) + 1;
        avg_range_end = std::min(avg_range_end, (int)data.size());

        int avg_range_length = avg_range_end - avg_range_start;
        for (; avg_range_start < avg_range_end; avg_range_start++) {
            avg_x += data[avg_range_start].x;
            avg_y += data[avg_range_start].y;
        }
        avg_x /= avg_range_length;
        avg_y /= avg_range_length;

        // Current bucket range
        int range_offs = (int)floor(i * bin_size) + 1;
        int range_to = (int)floor((i + 1) * bin_size) + 1;

        // Find point in current bucket that forms the largest triangle area
        double max_area = -1.0;
        for (; range_offs < range_to; range_offs++) {
            // Area formula: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            double area = std::abs(
                (data[a].x - avg_x) * (data[range_offs].y - data[a].y) -
                (data[a].x - data[range_offs].x) * (avg_y - data[a].y)
            ) * 0.5;

            if (area > max_area) {
                max_area = area;
                next_a = range_offs;
            }
        }

        sampled.push_back(data[next_a]);
        a = next_a; // Move to the next point
    }

    // Always include the last point
    sampled.push_back(data.back());
    return sampled;
}

// --- 3. The MinMax Pre-selection Wrapper ---
inline std::vector<Point> MinMaxLTTB(const std::vector<Point>& data, int threshold, int pre_ratio = 4) {
    int intermediate_size = threshold * pre_ratio;
    
    if (intermediate_size >= data.size()) {
        return LTTB(data, threshold);
    }

    // --- Stage 1: MinMax Pre-selection ---
    std::vector<Point> pre_sampled;
    pre_sampled.reserve(intermediate_size * 2); // Max 2 points per bucket
    
    struct BucketResult { int min_idx; int max_idx; };
    std::vector<BucketResult> results(intermediate_size);

    int bucket_size = data.size() / intermediate_size;

    #ifdef _OPENMP
    static bool once = true;
    if (once) {
        #pragma omp parallel
        #pragma omp single
        std::cout << "[MinMaxLTTB] OpenMP Active: " << omp_get_num_threads() << " threads." << std::endl;
        once = false;
    }
    #endif

    #pragma omp parallel for
    for (int i = 0; i < intermediate_size; ++i) {
        int start = i * bucket_size;
        int end = (i == intermediate_size - 1) ? data.size() : (i + 1) * bucket_size;
        
        int min_idx = start;
        int max_idx = start;
        double min_val = data[start].y;
        double max_val = data[start].y;

        int j = start + 1;

#ifdef __AVX2__
        // AVX2 Optimization: Process 4 points at a time
        __m256d v_min_val = _mm256_set1_pd(min_val);
        __m256d v_max_val = _mm256_set1_pd(max_val);
        __m256i v_min_idx = _mm256_set1_epi64x(min_idx);
        __m256i v_max_idx = _mm256_set1_epi64x(max_idx);
        __m256i v_curr_idx = _mm256_set_epi64x(j+3, j+2, j+1, j);
        __m256i v_inc = _mm256_set1_epi64x(4);

        for (; j <= end - 4; j += 4) {
            // Load 4 Points (x0 y0 x1 y1 | x2 y2 x3 y3)
            __m256d v_data1 = _mm256_loadu_pd(&data[j].x);
            __m256d v_data2 = _mm256_loadu_pd(&data[j+2].x);

            // Shuffle to extract Y values: y0 y2 y1 y3 -> permute to y0 y1 y2 y3
            __m256d v_y = _mm256_permute4x64_pd(
                _mm256_shuffle_pd(v_data1, v_data2, 0xF), 
                _MM_SHUFFLE(3, 1, 2, 0)
            );

            // Update Min
            __m256d mask_min = _mm256_cmp_pd(v_y, v_min_val, _CMP_LT_OQ);
            v_min_val = _mm256_min_pd(v_min_val, v_y);
            v_min_idx = _mm256_blendv_epi8(v_min_idx, v_curr_idx, _mm256_castpd_si256(mask_min));

            // Update Max
            __m256d mask_max = _mm256_cmp_pd(v_y, v_max_val, _CMP_GT_OQ);
            v_max_val = _mm256_max_pd(v_max_val, v_y);
            v_max_idx = _mm256_blendv_epi8(v_max_idx, v_curr_idx, _mm256_castpd_si256(mask_max));

            v_curr_idx = _mm256_add_epi64(v_curr_idx, v_inc);
        }

        // Reduce AVX results
        double tmp_min_v[4], tmp_max_v[4];
        long long tmp_min_i[4], tmp_max_i[4];
        _mm256_storeu_pd(tmp_min_v, v_min_val);
        _mm256_storeu_pd(tmp_max_v, v_max_val);
        _mm256_storeu_si256((__m256i*)tmp_min_i, v_min_idx);
        _mm256_storeu_si256((__m256i*)tmp_max_i, v_max_idx);

        for (int k = 0; k < 4; ++k) {
            if (tmp_min_v[k] < min_val) { min_val = tmp_min_v[k]; min_idx = (int)tmp_min_i[k]; }
            if (tmp_max_v[k] > max_val) { max_val = tmp_max_v[k]; max_idx = (int)tmp_max_i[k]; }
        }
#endif

        for (; j < end; ++j) {
            double val = data[j].y;
            if (val < min_val) { min_idx = j; min_val = val; }
            if (val > max_val) { max_idx = j; max_val = val; }
        }
        
        results[i] = {min_idx, max_idx};
    }

    for (const auto& res : results) {
        // Add min and max in chronological order
        if (res.min_idx < res.max_idx) { 
            pre_sampled.push_back(data[res.min_idx]); 
            pre_sampled.push_back(data[res.max_idx]); 
        } else if (res.min_idx > res.max_idx) { 
            pre_sampled.push_back(data[res.max_idx]); 
            pre_sampled.push_back(data[res.min_idx]); 
        } else {
            pre_sampled.push_back(data[res.min_idx]);
        }
    }

    // --- Stage 2: Standard LTTB on the reduced set ---
    return LTTB(pre_sampled, threshold);
}