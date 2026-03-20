#include <absl/container/flat_hash_map.h>
#include <spdlog/spdlog.h>

#include <optional>
#include <queue>

#include "bpe.hh"

namespace tokenizer {

std::vector<Delta> MergeWordPair(std::vector<uint32_t>& ids,
                                 TokenPair target_pair,
                                 uint32_t new_id) {
    std::vector<Delta> deltas;
    std::vector<uint32_t> out;
    out.reserve(ids.size());

    size_t i = 0;
    while (i < ids.size()) {
        if (i + 1 < ids.size()) {
            TokenPair pair(ids[i], ids[i + 1]);

            if (pair == target_pair) {
                std::optional<uint32_t> left = out.empty() ? std::nullopt : std::make_optional(out.back());
                std::optional<uint32_t> right = (i + 2 < ids.size()) ? std::make_optional(ids[i + 2]) : std::nullopt;

                if (left) {
                    deltas.push_back({TokenPair(*left, ids[i]), -1});
                    deltas.push_back({TokenPair(*left, new_id), 1});
                }

                deltas.push_back({pair, -1});

                if (right) {
                    deltas.push_back({TokenPair(ids[i + 1], *right), -1});
                    deltas.push_back({TokenPair(new_id, *right), 1});
                }

                out.push_back(new_id);
                i += 2;
                continue;
            }
        }
        out.push_back(ids[i]);
        i++;
    }

    ids = std::move(out);
    return deltas;
}

absl::flat_hash_map<uint32_t,
                    uint32_t>
TrainBPE(std::vector<std::vector<uint32_t>>& words,
         const std::vector<int>& counts,
         int num_merges) {
    absl::flat_hash_map<uint32_t, uint32_t> merges;
    if (num_merges <= 0) return merges;

    absl::flat_hash_map<uint64_t, uint64_t> pair_counts;
    absl::flat_hash_map<uint64_t, std::vector<size_t>> where_to_update;

    for (size_t i = 0; i < words.size(); ++i) {
        const auto& word = words[i];
        if (word.size() < 2 || counts[i] == 0) continue;

        for (size_t j = 1; j < word.size(); ++j) {
            auto pair = TokenPair(word[j - 1], word[j]);
            pair_counts[pair] += counts[i];
            where_to_update[pair].push_back(i);
        }
    }

    std::priority_queue<MergeJob> heap;
    for (const auto& [pair, pos] : where_to_update) {
        heap.push(MergeJob{pair, pair_counts[pair], pos});
    }

    int merges_done = 0;
    int last_log_percent = 0;

    while (merges_done < num_merges) {
        if (heap.empty()) break;

        auto top = heap.top();
        heap.pop();

        auto current = pair_counts[top.pair];
        if (current <= 0) continue;

        if (top.count != current) {
            top.count = current;
            heap.push(top);
            continue;
        }

        auto new_id = merges_done + 256;
        merges[top.pair] = new_id;

        absl::flat_hash_map<uint32_t, std::vector<size_t>> local_pos_updates;

        for (const auto word_idx : top.pos) {
            auto& ids = words[word_idx];
            auto deltas = MergeWordPair(ids, top.pair, new_id);

            for (const auto& delta : deltas) {
                auto delta_total = static_cast<int>(delta.delta) * counts[word_idx];

                if (delta_total != 0) {
                    pair_counts[delta.pair] += delta_total;

                    if (static_cast<int>(delta.delta) > 0) {
                        local_pos_updates[delta.pair].push_back(word_idx);
                    }
                }
            }
        }

        for (const auto& [pair, pos] : local_pos_updates) {
            auto cnt = pair_counts[pair];

            if (cnt > 0) {
                heap.push({pair, cnt, pos});
            }
        }
        merges_done += 1;

        auto cur_percent = (merges_done * 100) / num_merges;
        if (cur_percent > last_log_percent) {
            spdlog::info(
                "progress: {}% ({}/{} merges) - last merge: {} -> {} "
                "(frequency: {})",
                cur_percent,
                merges_done,
                num_merges,
                top.pair,
                new_id,
                top.count);
            last_log_percent = cur_percent;
        }
    }

    return merges;
}

}  // namespace tokenizer
