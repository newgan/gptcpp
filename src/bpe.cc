#include <queue>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <spdlog/spdlog.h>

#include "bpe.hh"

namespace tokenizer
{
    std::vector<Delta>
    MergeWordPair(std::vector<uint32_t>& chunk, TokenPair pair_to_merge, uint32_t new_vocab_id)
    {
        if (chunk.size() < 2)
        {
            return {};
        }

        auto last_token = chunk[0];
        std::vector<Delta> token_updates;
        std::vector<uint32_t> out_chunk;
        out_chunk.reserve(chunk.size());
        out_chunk.push_back(last_token);

        size_t chunk_size = chunk.size();

        for (size_t i = 1; i < chunk_size; ++i)
        {
            auto pair = TokenPair{last_token, chunk[i]};

            if (pair == pair_to_merge)
            {
                token_updates.push_back(Delta{pair, -1});

                if (out_chunk.size() > 1)
                {
                    auto left_token = out_chunk[out_chunk.size() - 2];
                    token_updates.push_back(Delta{TokenPair{left_token, out_chunk.back()}, -1});
                    token_updates.push_back(Delta{TokenPair{left_token, new_vocab_id}, 1});
                }
                if (i < chunk_size - 1)
                {
                    auto right_token = chunk[i + 1];
                    token_updates.push_back(Delta{TokenPair{chunk[i], right_token}, -1});
                    token_updates.push_back(Delta{TokenPair{new_vocab_id, right_token}, 1});
                }

                out_chunk.pop_back();
                out_chunk.push_back(new_vocab_id);
                last_token = new_vocab_id;
            }
            else
            {
                out_chunk.push_back(chunk[i]);
                last_token = chunk[i];
            }
        }
        chunk = std::move(out_chunk);
        return token_updates;
    }

    absl::flat_hash_map<TokenPair, uint32_t>
    TrainBPE(std::vector<std::vector<uint32_t>>& chunks, const std::vector<int>& chunk_counts, int num_merges)
    {
        absl::flat_hash_map<TokenPair, uint32_t> merges;

        size_t num_chunks = chunks.size();

        if (num_merges <= 0)
        {
            return merges;
        }

        absl::flat_hash_map<TokenPair, int64_t> num_pairs_in_chunks;

        absl::flat_hash_map<TokenPair, absl::flat_hash_set<size_t>> chunk_list_to_update;

        for (size_t i = 0; i < num_chunks; ++i)
        {
            const auto& word = chunks[i];
            if (word.size() < 2 || chunk_counts[i] == 0)
            {
                continue;
            }

            for (size_t j = 1; j < word.size(); ++j)
            {
                auto pair = TokenPair(word[j - 1], word[j]);
                num_pairs_in_chunks[pair] += chunk_counts[i];
                chunk_list_to_update[pair].insert(i);
            }
        }

        std::priority_queue<MergeJob> heap;
        for (const auto& [pair, count] : num_pairs_in_chunks)
        {
            heap.push(MergeJob{pair, count});
        }

        int merges_done = 0;
        int last_log_percent = 0;

        while (merges_done < num_merges)
        {
            if (heap.empty())
            {
                break;
            }

            auto top = heap.top();
            heap.pop();

            int64_t current_true_count = num_pairs_in_chunks[top.pair];
            if (current_true_count <= 0 || top.count != current_true_count)
            {
                continue;
            }

            auto next_vocab_id = merges_done + 256;
            merges[top.pair] = next_vocab_id;

            absl::flat_hash_set<TokenPair> pairs_to_recalculate;

            const auto positions = chunk_list_to_update[top.pair];

            for (const auto chunk_position : positions)
            {
                auto& chunk = chunks[chunk_position];
                const auto& updates = MergeWordPair(chunk, top.pair, next_vocab_id);

                for (const auto& delta_obj : updates)
                {
                    int64_t total_delta = static_cast<int64_t>(delta_obj.delta) * chunk_counts[chunk_position];

                    if (total_delta != 0)
                    {
                        num_pairs_in_chunks[delta_obj.pair] += total_delta;
                        pairs_to_recalculate.insert(delta_obj.pair);

                        if (delta_obj.delta > 0)
                        {
                            chunk_list_to_update[delta_obj.pair].insert(chunk_position);
                        }
                    }
                }
            }

            for (const auto& updated_pair : pairs_to_recalculate)
            {
                int64_t updated_count = num_pairs_in_chunks[updated_pair];
                if (updated_count > 0)
                {
                    heap.push(MergeJob{updated_pair, updated_count});
                }
            }

            chunk_list_to_update.erase(top.pair);

            merges_done++;

            auto cur_percent = (merges_done * 100) / num_merges;
            if (cur_percent > last_log_percent)
            {
                spdlog::info(
                    "progress: {}% ({}/{} merges) - frequency: {}",
                    cur_percent,
                    merges_done,
                    num_merges,
                    current_true_count
                );
                last_log_percent = cur_percent;
            }
        }

        return merges;
    }
}  // namespace tokenizer
