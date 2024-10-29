#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <thread>  // For sleep
namespace fs = std::filesystem;

int main() {
    // Start date: 2023-01-01
    std::tm start_date = {};
    start_date.tm_year = 2023 - 1900;  // Years since 1900
    start_date.tm_mon = 0;             // January (0-indexed)
    start_date.tm_mday = 1;            // Day 1

    // End date: 2023-12-31
    std::tm end_date = {};
    end_date.tm_year = 2023 - 1900;
    end_date.tm_mon = 11;              // December (0-indexed)
    end_date.tm_mday = 31;

    // Convert start and end dates to time points
    auto start = std::chrono::system_clock::from_time_t(std::mktime(&start_date));
    auto end = std::chrono::system_clock::from_time_t(std::mktime(&end_date));

    // Iterate day by day
    for (auto current = start; current <= end; current += std::chrono::hours(24)) {
        // Convert to std::tm for printing
        std::time_t current_time = std::chrono::system_clock::to_time_t(current);
        std::tm* date = std::localtime(&current_time);

        // Print date in "YYYY_MM_DD" format
        std::ostringstream oss;
        oss << std::put_time(date, "%Y_%m_%d");
        std::string date_str = oss.str();
        std::string filename = "sorted_data/AIS_" + date_str + ".csv";
        if (fs::exists(filename)) {
            std::ifstream       file(filename);

        } else {
            std::this_thread::sleep_for(std::chrono::seconds(10));

            // Check one more time after waiting
            if (fs::exists(filename)) {
                std::ifstream       file(filename);
            }

        }

        std::cout << std::put_time(date, "%Y_%m_%d") << std::endl;
    }

    return 0;
}
