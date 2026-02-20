#include <GLFW/glfw3.h>
#include <omp.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include "algorithms.h"

#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <iostream>

// --- Data Generation Helpers ---
std::vector<Point> GenerateSineNoise(int count) {
    std::vector<Point> data(count);
    std::mt19937 gen(42);
    std::normal_distribution<> d(0, 0.5); // Noise
    for (int i = 0; i < count; ++i) {
        double t = (double)i / count * 20.0; // 20 cycles
        data[i] = { (double)i, std::sin(t) * 10.0 + d(gen) };
    }
    return data;
}

std::vector<Point> GenerateBrownian(int count) {
    std::vector<Point> data(count);
    std::mt19937 gen(42);
    std::normal_distribution<> d(0, 1.0);
    double y = 0;
    for (int i = 0; i < count; ++i) {
        y += d(gen);
        data[i] = { (double)i, y };
    }
    return data;
}

std::vector<Point> GenerateImpulse(int count) {
    std::vector<Point> data(count);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, count - 1);
    
    for (int i = 0; i < count; ++i) data[i] = { (double)i, 0.0 };
    
    // Add 50 random spikes
    for (int i = 0; i < 50; ++i) {
        int idx = dist(gen);
        data[idx].y = 100.0;
    }
    return data;
}

// --- Application State ---
enum Algorithm { ALG_RAW, ALG_LTTB, ALG_MINMAX };
enum DataType { DATA_SINE, DATA_BROWNIAN, DATA_IMPULSE };

struct AppState {
    // Data
    std::vector<Point> rawData;
    std::vector<Point> downsampledData;
    
    // Parameters
    int targetResolution = 1000;
    int preSelectionRatio = 4;
    int threadCount = 1;
    Algorithm currentAlgo = ALG_LTTB;
    DataType currentDataType = DATA_SINE;
    bool showRaw = false;
    
    // Metrics
    double lastExecTimeUs = 0.0;
    double benchmarkLttbAvg = 0.0;
    double benchmarkMinMaxAvg = 0.0;
    
    // Flags
    bool dataDirty = true;
    bool algoDirty = true;
};

void UpdateData(AppState& state) {
    const int N = 1000000;
    if (state.currentDataType == DATA_SINE) state.rawData = GenerateSineNoise(N);
    else if (state.currentDataType == DATA_BROWNIAN) state.rawData = GenerateBrownian(N);
    else if (state.currentDataType == DATA_IMPULSE) state.rawData = GenerateImpulse(N);
    state.dataDirty = false;
    state.algoDirty = true;
}

void RunAlgorithm(AppState& state) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (state.currentAlgo == ALG_RAW) {
        // No op, just point to raw (or copy if needed for consistency, but we handle rendering separately)
        state.downsampledData.clear(); 
    } else if (state.currentAlgo == ALG_LTTB) {
        state.downsampledData = LTTB(state.rawData, state.targetResolution);
    } else if (state.currentAlgo == ALG_MINMAX) {
        state.downsampledData = MinMaxLTTB(state.rawData, state.targetResolution, state.preSelectionRatio);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    state.lastExecTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    state.algoDirty = false;
}

void RunBenchmark(AppState& state) {
    // Benchmark LTTB
    double totalLttb = 0;
    for(int i=0; i<100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        volatile auto res = LTTB(state.rawData, state.targetResolution);
        auto end = std::chrono::high_resolution_clock::now();
        totalLttb += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    state.benchmarkLttbAvg = totalLttb / 100.0;

    // Benchmark MinMax
    double totalMinMax = 0;
    for(int i=0; i<100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        volatile auto res = MinMaxLTTB(state.rawData, state.targetResolution, state.preSelectionRatio);
        auto end = std::chrono::high_resolution_clock::now();
        totalMinMax += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    state.benchmarkMinMaxAvg = totalMinMax / 100.0;

    std::cout << "Benchmark Results (Avg):" << std::endl;
    const char* dataNames[] = { "Sine + Noise", "Brownian Motion", "Impulse Spikes" };
    std::cout << "Dataset: " << dataNames[state.currentDataType] << std::endl;
    std::cout << "LTTB: " << state.benchmarkLttbAvg << " us" << std::endl;
    std::cout << "MinMaxLTTB: " << state.benchmarkMinMaxAvg << " us" << std::endl;
    std::cout << "Speedup: " << state.benchmarkLttbAvg / state.benchmarkMinMaxAvg << "x" << std::endl;
}

int main(int, char**) {
    // Setup Window
    if (!glfwInit()) return 1;
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "LTTB vs MinMaxLTTB Evaluator", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    AppState state;
    state.threadCount = omp_get_max_threads();
    UpdateData(state);

    // Main Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Logic Update
        if (state.dataDirty) UpdateData(state);
        if (state.algoDirty) RunAlgorithm(state);

        // Start Frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        // --- Dashboard ---
        ImGui::Begin("Control Panel");
        
        ImGui::Text("Dataset Configuration");
        if (ImGui::Combo("Data Type", (int*)&state.currentDataType, "Sine + Noise\0Brownian Motion\0Impulse Spikes\0")) {
            state.dataDirty = true;
        }
        ImGui::Separator();

        ImGui::Text("Algorithm Settings");
        if (ImGui::Combo("Algorithm", (int*)&state.currentAlgo, "Raw Data\0LTTB\0MinMaxLTTB\0")) {
            state.algoDirty = true;
        }

        if (ImGui::SliderInt("Resolution (Points)", &state.targetResolution, 100, 10000)) {
            state.algoDirty = true;
        }

        if (state.currentAlgo == ALG_MINMAX) {
            if (ImGui::SliderInt("Pre-selection Ratio", &state.preSelectionRatio, 2, 32)) {
                state.algoDirty = true;
            }
            if (ImGui::SliderInt("OpenMP Threads", &state.threadCount, 1, omp_get_num_procs())) {
                omp_set_num_threads(state.threadCount);
                state.algoDirty = true;
            }
        }

        ImGui::Separator();
        ImGui::Checkbox("Overlay Raw Data (Gray)", &state.showRaw);

        ImGui::Separator();
        if (ImGui::Button("Run Benchmark (100 iters)")) {
            RunBenchmark(state);
        }

        // Stats
        ImGui::Separator();
        ImGui::Text("Current Frame Stats:");
        ImGui::Text("Input Size: %zu", state.rawData.size());
        if (state.currentAlgo != ALG_RAW) {
            ImGui::Text("Output Size: %zu", state.downsampledData.size());
            ImGui::Text("Reduction: %.1fx", (double)state.rawData.size() / state.downsampledData.size());
            ImGui::Text("Time: %.2f us", state.lastExecTimeUs);
        } else {
            ImGui::Text("Output Size: %zu", state.rawData.size());
        }

        if (state.benchmarkLttbAvg > 0) {
            ImGui::Separator();
            const char* dataNames[] = { "Sine + Noise", "Brownian Motion", "Impulse Spikes" };
            ImGui::Text("Benchmark Results (Avg) [%s]:", dataNames[state.currentDataType]);
            ImGui::Text("LTTB: %.2f us", state.benchmarkLttbAvg);
            ImGui::Text("MinMaxLTTB: %.2f us", state.benchmarkMinMaxAvg);
            double speedup = state.benchmarkLttbAvg / state.benchmarkMinMaxAvg;
            ImGui::TextColored(ImVec4(0,1,0,1), "Speedup: %.2fx", speedup);
        }

        ImGui::End();

        // --- Visualization ---
        ImGui::Begin("Visualization");
        if (ImPlot::BeginPlot("Signal", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("Index", "Value");
            
            // Optional Raw Data Overlay
            if (state.showRaw || state.currentAlgo == ALG_RAW) {
                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f);
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f, 0.5f, 0.5f, 0.4f)); // Gray, transparent
                ImPlot::PlotLine("Raw Data", &state.rawData[0].x, &state.rawData[0].y, state.rawData.size(), 0, 0, sizeof(Point));
                ImPlot::PopStyleColor();
                ImPlot::PopStyleVar();
            }

            // Downsampled Data
            if (state.currentAlgo != ALG_RAW && !state.downsampledData.empty()) {
                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.7f, 1.0f, 1.0f)); // High contrast Blue
                ImPlot::PlotLine("Downsampled", &state.downsampledData[0].x, &state.downsampledData[0].y, state.downsampledData.size(), 0, 0, sizeof(Point));
                ImPlot::PopStyleColor();
                ImPlot::PopStyleVar();
            }

            ImPlot::EndPlot();
        }
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}