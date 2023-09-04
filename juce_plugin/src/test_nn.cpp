#include "DiodeClipper.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <random>


std::vector<float> randomVector(std::size_t size) {
    std::vector<float> result(size);
    
    // 使用现代C++随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd()); // 使用Mersenne Twister引擎
    std::uniform_real_distribution<> dis(-1.0, 1.0); // 生成-1到1之间的随机数

    for (std::size_t i = 0; i < size; ++i) {
        result[i] = dis(gen);
    }

    return result;
}

int main() {
    DiodeClipper dc{48000.0f};
    
    std::vector<float> v_in = randomVector(1024);
    
//    dc.forward(v_in, 50e3f);
    
    return 0;
}
