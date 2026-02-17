
Evaluate the performance and visual fidelity of **LTTB** vs. **MinMaxLTTB**

Provide a clean, modular environment that separates data generation, downsampling logic, and the UI/rendering loop.

Here is a comprehensive requirements list.

---

## 1. Technical Stack & Dependencies

The project must be built for **Linux** using **CMake** (v3.21+) and target **OpenGL 3** for rendering.

### CMake FetchContent Requirements

The developer must use `FetchContent` to manage dependencies. This ensures a "one-click" build without manual installation of libraries.

* **Dear ImGui**: Use the `docking` branch. Require the `imgui_impl_glfw` and `imgui_impl_opengl3` backends.
* **ImPlot**: Fetched as a dependency of ImGui or standalone.
* **GLFW3**: For window and context management.

### Downsampling Logic

* **LTTB Implementation**: Should be a header-only or simple utility class (e.g., based on the Steinarsson reference).
* **MinMaxLTTB Implementation**: A two-stage algorithm:
1. **Stage 1 (MinMax Pre-selection)**: Divide the input into  buckets (where  is the pre-selection ratio, typically 2 or 4). Keep the min and max of each bucket.
2. **Stage 2 (LTTB)**: Run the standard LTTB on the significantly reduced dataset from Stage 1.



---

## 2. Functional Requirements

### Dashboard & Control Panel

The ImGui interface must allow the user to:

* **Select Algorithm**: Toggle between "Raw Data," "LTTB," and "MinMaxLTTB."
* **Resolution Control**: A slider to set the number of output points ().
* **Pre-selection Ratio ()**: A slider (specific to MinMaxLTTB) to adjust the aggressiveness of the first pass.
* **Trigger Benchmark**: A button to run 100 iterations of each algorithm and report the average time.

### Visualization (ImPlot)

* **Overlay Mode**: Display the original signal (optional/grayed out) and the downsampled signal (high contrast) simultaneously to check for visual artifacts (e.g., aliasing or missed peaks).
* **Performance Metrics Overlay**: Display a small text box or "Stats" window showing:
* Execution time in microseconds ().
* Reduction ratio (e.g., 1,000,000  1,000 points).

---

## 3. Representative Dataset Proposal

To truly test these algorithms, you need data with varying "noise-to-signal" ratios. A dataset of **1,000,000 points** is recommended.

| Data Type | Description | Purpose |
| --- | --- | --- |
| **Synthetic Sine + Noise** | A low-frequency sine wave with high-frequency Gaussian noise. | Tests peak preservation and "triangle area" logic. |
| **Brownian Motion** | A random walk signal. | Tests how algorithms handle "wandering" data without clear peaks. |
| **Impulse Spikes** | Flat signal with extremely narrow, tall spikes (1-2 points wide). | **Crucial:** Checks if downsampling "misses" critical anomalies (aliasing). |

---

## 4. Implementation Details for the Dev

> **Note to Dev:** Please ensure the benchmarking loop uses `std::chrono::high_resolution_clock`. For the OpenGL setup on Linux, ensure `libgl1-mesa-dev` and `xorg-dev` are listed as system requirements in the README.

### Example CMake Snippet

```cmake
include(FetchContent)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        docking
)
FetchContent_MakeAvailable(imgui)

# Repeat for implot, glfw, etc.

```

To give you a head start, I've outlined a high-performance C++ skeleton. This design uses `std::vector` for memory safety and keeps the logic decoupled from the ImGui rendering loop.

The key to **MinMaxLTTB** is that it performs a "cheap" pass (simple comparisons) to prune the data before the "expensive" pass (triangle area calculations).

---

## 1. Data Structure

First, define a simple point structure that plays nice with ImPlot's expectation of contiguous memory.

```cpp
struct Point {
    double x, y;
};

```

---

## 2. The LTTB Logic (Largest Triangle Three Buckets)

This is the "gold standard" for visual downsampling. It ensures that the visual characteristics (peaks and valleys) of the line are preserved.

```cpp
std::vector<Point> LTTB(const std::vector<Point>& data, int threshold) {
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
        int avg_range_start = floor((i + 1) * bin_size) + 1;
        int avg_range_end = floor((i + 2) * bin_size) + 1;
        avg_range_end = std::min(avg_range_end, (int)data.size());

        int avg_range_length = avg_range_end - avg_range_start;
        for (; avg_range_start < avg_range_end; avg_range_start++) {
            avg_x += data[avg_range_start].x;
            avg_y += data[avg_range_start].y;
        }
        avg_x /= avg_range_length;
        avg_y /= avg_range_length;

        // Current bucket range
        int range_offs = floor(i * bin_size) + 1;
        int range_to = floor((i + 1) * bin_size) + 1;

        // Find point in current bucket that forms the largest triangle area
        double max_area = -1.0;
        for (; range_offs < range_to; range_offs++) {
            // Area formula: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            double area = fabs(
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

```

---

## 3. The MinMax Pre-selection Wrapper

MinMaxLTTB is essentially LTTB with a "coarse filter" applied first. This significantly speeds up processing for massive datasets (e.g.,  points) because it reduces the number of triangle area calculations.

```cpp
std::vector<Point> MinMaxLTTB(const std::vector<Point>& data, int threshold, int pre_ratio = 4) {
    int intermediate_size = threshold * pre_ratio;
    
    if (intermediate_size >= data.size()) {
        return LTTB(data, threshold);
    }

    // --- Stage 1: MinMax Pre-selection ---
    std::vector<Point> pre_sampled;
    pre_sampled.reserve(intermediate_size);
    
    int bucket_size = data.size() / intermediate_size;
    for (int i = 0; i < intermediate_size; ++i) {
        int start = i * bucket_size;
        int end = (i == intermediate_size - 1) ? data.size() : (i + 1) * bucket_size;
        
        int min_idx = start, max_idx = start;
        for (int j = start + 1; j < end; ++j) {
            if (data[j].y < data[min_idx].y) min_idx = j;
            if (data[j].y > data[max_idx].y) max_idx = j;
        }
        
        // Add min and max in chronological order
        if (min_idx < max_idx) {
            pre_sampled.push_back(data[min_idx]);
            pre_sampled.push_back(data[max_idx]);
        } else {
            pre_sampled.push_back(data[max_idx]);
            pre_sampled.push_back(data[min_idx]);
        }
    }

    // --- Stage 2: Standard LTTB on the reduced set ---
    return LTTB(pre_sampled, threshold);
}

```

---

## 4. Suggested Implementation Next Steps

Tell your developer to focus on these three things once the skeleton is in place:

1. **Cache the results**: Don't re-run the algorithm every frame. Only re-run if the `threshold` or `data` changes.
2. **ImPlot Integration**: Use `ImPlot::PlotLine` by passing the address of the first element:
```cpp
// Inside the ImGui Render Loop
if (ImPlot::BeginPlot("Performance Test")) {
    ImPlot::PlotLine("Downsampled", &sampled[0].x, &sampled[0].y, sampled.size(), 0, 0, sizeof(Point));
    ImPlot::EndPlot();
}

```


3. **SIMD Optimization**: If performance is still a bottleneck, suggest they look into SIMD (SSE/AVX) for the MinMax pre-selection stage, as it is a classic "search" problem.
