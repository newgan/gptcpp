#ifndef GPTCPP_BPE_H_
#define GPTCPP_BPE_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <boost/regex/v5/regex_fwd.hpp>

#include "tokenizer.hh"

namespace tokenizer
{
    struct MergeJob
    {
        TokenPair pair;
        int64_t count;

        bool operator<(const MergeJob& other) const
        {
            auto p1 = count;
            auto p2 = other.count;
            if (p1 != p2)
            {
                return p1 < p2;
            }
            return pair > other.pair;
        }
    };

    struct Delta
    {
        TokenPair pair;
        int delta;
    };

    absl::flat_hash_map<std::string, int>
    SplitAndCountChunks(const std::vector<std::filesystem::path>& paths, const boost::regex& pattern);
    std::vector<Delta>
    MergeWordPair(std::vector<uint32_t>& chunk, TokenPair pair_to_merge, uint32_t new_vocab_id);

    absl::flat_hash_map<TokenPair, uint32_t>
    TrainBPE(std::vector<std::vector<uint32_t>>& chunks, const std::vector<int>& chunk_counts, int num_merges);

}  // namespace tokenizer

#endif
