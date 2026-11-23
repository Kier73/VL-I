// A row program assigns each original BF16 coefficient an integer class.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace {
using Code = std::uint16_t;
struct Plan {
    std::size_t rows, cols;
    std::vector<Code> index, codes;
    std::vector<std::size_t> offsets;
};

// decode(b) = IEEE754_binary32(b << 16), converted exactly to binary64.
double decode(Code b) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}

// b_j = byte[2j] + 256 byte[2j+1].
Code word(const unsigned char* source, std::size_t j) {
    return source[2*j] | (static_cast<Code>(source[2*j+1]) << 8);
}
}

// Form I_ij=k and B_ik=b iff bits(W_ij)=b, with k ordered by first occurrence.
extern "C" void* form(const unsigned char* data, std::size_t bytes,
                      std::size_t rows, std::size_t cols) noexcept {
    if (!data || !rows || !cols || bytes/2/rows != cols || bytes != 2*rows*cols)
        return nullptr;
    try {
        auto p = std::make_unique<Plan>();
        p->rows = rows;
        p->cols = cols;
        p->index.resize(rows*cols);
        std::vector<int> ordinal(65536, -1);
        for (std::size_t i = 0; i < rows; ++i) {
            const auto start = p->codes.size();
            p->offsets.push_back(start);
            for (std::size_t j = i*cols; j < (i+1)*cols; ++j) {
                Code b = word(data,j);
                if (ordinal[b] < 0) {
                    if (!std::isfinite(decode(b))) return nullptr;
                    ordinal[b] = static_cast<int>(p->codes.size()-start);
                    p->codes.push_back(b);
                }
                p->index[j] = static_cast<Code>(ordinal[b]);
            }
            for (auto k = start; k < p->codes.size(); ++k) ordinal[p->codes[k]] = -1;
        }
        p->offsets.push_back(p->codes.size());
        return p.release();
    } catch (...) {
        return nullptr;
    }
}

// K = Σ_i |{W_ij : j in row i}|, counting distinct encodings.
extern "C" std::size_t classes(const Plan* p) { return p->codes.size(); }

// y_i = bias_i + Σ_k decode(B_ik) Σ_{j:I_ij=k} x_j.
extern "C" void interpret(const Plan* p, const double* x, double* y, double* sums) {
    for (std::size_t i = 0; i < p->rows; ++i) {
        auto start = p->offsets[i];
        auto count = p->offsets[i+1]-start;
        std::fill_n(sums, count, 0.);
        for (std::size_t j = 0; j < p->cols; ++j) sums[p->index[i*p->cols+j]] += x[j];
        double dot = 0.;
        for (std::size_t k = 0; k < count; ++k) dot += decode(p->codes[start+k])*sums[k];
        y[i] += dot;
    }
}

// y_i = bias_i + Σ_j decode(bits(W_ij)) x_j in original column order.
extern "C" void direct(const unsigned char* data, std::size_t rows, std::size_t cols,
                       const double* x, double* y) {
    for (std::size_t i = 0; i < rows; ++i) {
        double dot = 0.;
        for (std::size_t j = 0; j < cols; ++j) dot += decode(word(data,i*cols+j))*x[j];
        y[i] += dot;
    }
}

// Retained arrays for (I,B) transition from {index,codes,offsets} to ∅.
extern "C" void release(Plan* p) { delete p; }
