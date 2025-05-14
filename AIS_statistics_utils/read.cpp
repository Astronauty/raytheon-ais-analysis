#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "cnpy.h"  

#include <iomanip>
#include <chrono>
#include <filesystem>
#include <thread>  // For sleep

class CSVRow
{
    public:
        std::string_view operator[](std::size_t index) const
        {
            return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
        }
        std::size_t size() const
        {
            return m_data.size() - 1;
        }
        void readNextRow(std::istream& str)
        {
            std::getline(str, m_line);

            m_data.clear();
            m_data.emplace_back(-1);
            std::string::size_type pos = 0;
            while((pos = m_line.find(',', pos)) != std::string::npos)
            {
                m_data.emplace_back(pos);
                ++pos;
            }
            // This checks for a trailing comma with no data after it.
            pos   = m_line.size();
            m_data.emplace_back(pos);
        }
    private:
        std::string         m_line;
        std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}   

class CSVIterator
{   
    public:
        typedef std::input_iterator_tag     iterator_category;
        typedef CSVRow                      value_type;
        typedef std::size_t                 difference_type;
        typedef CSVRow*                     pointer;
        typedef CSVRow&                     reference;

        CSVIterator(std::istream& str)  :m_str(str.good()?&str:nullptr) { ++(*this); }
        CSVIterator()                   :m_str(nullptr) {}

        // Pre Increment
        CSVIterator& operator++()               {if (m_str) { if (!((*m_str) >> m_row)){m_str = nullptr;}}return *this;}
        // Post increment
        CSVIterator operator++(int)             {CSVIterator    tmp(*this);++(*this);return tmp;}
        CSVRow const& operator*()   const       {return m_row;}
        CSVRow const* operator->()  const       {return &m_row;}

        bool operator==(CSVIterator const& rhs) {return ((this == &rhs) || ((this->m_str == nullptr) && (rhs.m_str == nullptr)));}
        bool operator!=(CSVIterator const& rhs) {return !((*this) == rhs);}
    private:
        std::istream*       m_str;
        CSVRow              m_row;
};
class CSVRange
{
    std::istream&   stream;
    public:
        CSVRange(std::istream& str)
            : stream(str)
        {}
        CSVIterator begin() const {return CSVIterator{stream};}
        CSVIterator end()   const {return CSVIterator{};}
};
int get_seconds(std::string timestamp){
    // std::cout << timestamp<<std::endl;
    std::string hour_str = timestamp.substr(11, 2);
    std::string minute_str = timestamp.substr(14, 2);
    std::string second_str = timestamp.substr(17, 2);
    int hour = std::stoi(hour_str);
    int minute = std::stoi(minute_str);
    int second = std::stoi(second_str);
    int seconds = hour * 3600 + minute * 60 + second;
    return seconds;
}
void run_stat(std::string filename, std::string mfilename)
{   
    int M = 9;
    int N = 86400;
    std::vector<std::vector<int>> matrix(M, std::vector<int>(N, 0));
    
    std::ifstream       file(filename);
    int last_time = 0;
    std::string last_mmsi;
    bool first_parsed = false;
    bool parsed_first_row = false;
    int time = 0;
    std::string mmsi;
    int group;

    for(auto& row: CSVRange(file))
    {   
        try {
        if (!parsed_first_row){
            parsed_first_row = true;
            continue;
        }
        if (!first_parsed){
            last_mmsi = row[1];
            last_time = get_seconds(std::string(row[2]));
            first_parsed = true;
            group = std::stoi(std::string(row[3]));
        }
        else{
            mmsi = row[1];
            if (mmsi == last_mmsi){
                time = get_seconds(std::string(row[2]));
                int delta = time - last_time;
                matrix[group][delta] += 1;
                
            }
            else{
                last_mmsi = mmsi;
                last_time = get_seconds(std::string(row[2]));
                group = std::stoi(std::string(row[3]));
            }
        }
        
        } catch (const std::exception& e) {
            first_parsed = false;
            // Or, if within main, you can also simply use `return 1;`
            continue;
        }
        
    }

    std::vector<int> flat_matrix;
    for (const auto& row : matrix) {
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }

    // Define shape for 2D array (rows, columns)
    std::vector<size_t> shape = {static_cast<size_t>(M), static_cast<size_t>(N)};

    // Save the matrix to a .npy file
    cnpy::npy_save(mfilename, flat_matrix.data(), shape);

    // std::cout << "Matrix saved to matrix.npy" << std::endl;
}



int main() {
    // Start date: 2023-01-01
    std::tm start_date = {};
    start_date.tm_year = 2023 - 1900;  // Years since 1900
    start_date.tm_mon = 3;             // January (0-indexed)
    start_date.tm_mday = 27;            // Day 1

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
        std::string ofilename = "npys/" + date_str + ".npy";
        if (std::filesystem::exists(filename)) {
            std::ifstream       file(filename);
            run_stat(filename,  ofilename);
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            // Check one more time after waiting
            if (std::filesystem::exists(filename)) {
                std::ifstream       file(filename);
                run_stat(filename,  ofilename);
            }
        }
    }

    return 0;
}
