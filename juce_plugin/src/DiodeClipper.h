#pragma once

#include <array>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

enum DiodeType { DIODE_1N34A = 0, DIODE_1N4148 = 1 };
enum DiodeConfig { CONFIG_1U1D = 0, CONFIG_1U2D = 1 };

class DiodeClipper {
public:
    DiodeClipper(float fs_);
        
    void update_impedance();    // update impedance
    void set_fs(float fs_);     // set frequency sampling
    void set_vs_r(float vs_r_); // set resistor
    void set_model(DiodeType diode_, DiodeConfig config_);    // change diode model or diode pair configuration
    inline void reset();        // reset state
    float forward(float v_in);  // forward function
    
private:
    DiodeType diode {};         // diode model
    DiodeConfig config {};      // diode pair configuration
    float fs {48000.0f};        // sampling frequency
    const float c1 {4.7e-9f};   // capacitance of capacitor C1
    float c1_r;                 // impedance of capacitor C1
    float vs_r {50e3f};         // resistance of resistor R1
    float dp_r {};              // input impedance of diode pair
    float output {};            // output of diode pair
    
    std::array<float, 3> p1_r {};   // impedance of parallel adaptor P1
    std::array<float, 3> p1_a {};   // incident waves of parallel adaptor P1
    std::array<float, 3> p1_b {};   // reflected waves of parallel adaptor P1
    std::array<std::array<float, 3>, 3> p1; // scattering matrix of parallel adaptor P1
    
    const std::string basePath {"/Users/yangshijie/Projects/Capstone_2023/_L13/my_wdf_py/export_model/"};
    const std::string extension {".pt"};
    std::map<std::pair<DiodeType, DiodeConfig>, std::string> modelMappings;
    std::vector<torch::jit::IValue> input_tensor{torch::tensor({0.0f, 0.0f}, torch::dtype(torch::kFloat).requires_grad(false))};
    float* dp_in;
    
    torch::jit::script::Module dp;  // trained diode pair model
    
    bool load_model(const std::string& path);
};
