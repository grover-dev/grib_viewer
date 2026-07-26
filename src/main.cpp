#include <print>

#include <boatforge/version.hpp>

int main() {
    std::println("boatforge {} — hello, world", boatforge::kVersion);
    return 0;
}
