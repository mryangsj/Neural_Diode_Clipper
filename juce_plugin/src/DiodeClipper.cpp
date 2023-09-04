#include "DiodeClipper.h"


DiodeClipper::DiodeClipper(float fs_) : fs{fs_} {
    c1_r = 1 / (2 * c1 * fs);
    p1[0][0] = 0.0f;
    p1[1][0] = 1.0f;
    p1[2][0] = 1.0f;
    dp_in = input_tensor[0].toTensor().data_ptr<float>();
    
    modelMappings = {
        {{DIODE_1N34A, CONFIG_1U1D}, "1n34a_1u1d"},
        {{DIODE_1N34A, CONFIG_1U2D}, "1n34a_1u2d"},
        {{DIODE_1N4148, CONFIG_1U1D}, "1n4148_1u1d"},
        {{DIODE_1N4148, CONFIG_1U2D}, "1n4148_1u2d"}
    };
    
    update_impedance();
    set_model(DiodeType(DIODE_1N4148), DiodeConfig(CONFIG_1U1D));
}


void DiodeClipper::update_impedance() {
    //==============================================================================
    // update impedance of parallel adaptor
    p1_r[0] = (c1_r * vs_r) / (c1_r + vs_r);
    p1_r[1] = c1_r;
    p1_r[2] = vs_r;
    
    //==============================================================================
    // update scatering matrix
    float factor = p1_r[1] + p1_r[2];
    p1[0][1] = p1_r[2] / factor;
    p1[0][2] = p1_r[1] / factor;
    p1[1][1] = -p1[0][2];
    p1[1][2] = p1[0][2];
    p1[2][1] = p1[0][1];
    p1[2][2] = -p1[0][1];
    
    //==============================================================================
    // update impedance of diode pair
    dp_r = (p1_r[0] - 1.7e3f) / (2.5e3f - 1.7e3f);
}


void DiodeClipper::set_fs(float fs_) {
    fs = fs_;
    c1_r = 1 / (2 * c1 * fs);
    update_impedance();
}


void DiodeClipper::set_vs_r(float vs_r_) {
    vs_r = vs_r_;
    update_impedance();
}


inline void DiodeClipper::reset() {
    for (size_t i{}; i < 3; i++) {
        p1_a[i] = 0.0f;
        p1_b[i] = 0.0f;
    }
}


bool DiodeClipper::load_model(const std::string& path) {
    try {
        dp = torch::jit::load(path);
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model from: " << path << "\n";
        return false;
    }
}


void DiodeClipper::set_model(DiodeType diode_, DiodeConfig config_) {
    std::string modelName = modelMappings[{diode_, config_}];
    if (modelName.empty()) {
        modelName = "1n4148_1u1d"; // default
    }
    
    if (load_model(basePath + modelName + extension)) {
        diode = diode_;
        config = config_;
        dp.eval();
    } else {
        // Handle error or provide feedback.
    }
}



float DiodeClipper::forward(float v_in) {
    //==============================================================================
    // forward scan
    p1_a[1] = p1_b[1];
    p1_a[2] = v_in;
    p1_b[0] = p1[0][1]*p1_a[1] + p1[0][2]*p1_a[2];
    
    //==============================================================================
    // local root scattering
    dp_in[0] = p1_b[0];
    dp_in[1] = dp_r;
    
    {
        torch::NoGradGuard no_grad;
        p1_a[0] = dp.forward(input_tensor).toTensor().item<float>();
    }
    
    //==============================================================================
    // backward scan
    p1_b[1] = p1[1][0]*p1_a[0] + p1[1][1]*p1_a[1] + p1[1][2]*p1_a[2];
    p1_b[2] = p1[2][0]*p1_a[0] + p1[2][1]*p1_a[1] + p1[2][2]*p1_a[2];
    
    //==============================================================================
    // read output
    output = (p1_a[0] + p1_b[0]) / 2.0f;
    return output;
}
