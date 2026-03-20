#ifndef NANOGPTCPP_BPE_H_
#define NANOGPTCPP_BPE_H_

#include <absl/container/flat_hash_map.h>

#include <cassert>
#include <cstdint>
#include <vector>

namespace tokenizer {
class TokenPair {
   public:
    TokenPair() = default;

    constexpr TokenPair(uint32_t left,
                        uint32_t right) {
        assert(left <= 0xFFFF && "token ID exceeds 32-bit limit");
        assert(right <= 0xFFFF && "token ID exceeds 32-bit limit");

        pair_ = (static_cast<uint64_t>(left) << 32) | right;
    }

    constexpr auto operator<=>(const TokenPair& other) const noexcept {
        return pair_ <=> other.pair_;
    }

    uint32_t left() const { return pair_ >> 32; }
    uint32_t right() const { return pair_ & 0xFFFF; }
    uint64_t value() const { return pair_; }

   private:
    uint64_t pair_{};
};

struct MergeJob {
    TokenPair pair;
    uint64_t count;
    std::vector<size_t> pos;
    bool operator<(const MergeJob& other) const {
        if (count != other.count) return count < other.count;
        return pair > other.pair;
    }
};

struct Delta {
    TokenPair pair;
    int delta;
};

std::vector<Delta> MergeWordPair(std::vector<uint32_t>& ids,
                                 uint32_t target_pair,
                                 uint32_t new_id);

absl::flat_hash_map<uint32_t,
                    uint32_t>
TrainBPE(std::vector<std::vector<uint32_t>>& words,
         const std::vector<int>& counts,
         int num_merges);

}  // namespace tokenizer

#endif
