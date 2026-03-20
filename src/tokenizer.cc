#include <absl/container/flat_hash_map.h>

#include <boost/regex.hpp>
#include <cassert>
#include <fstream>
#include <queue>
#include <string>

#include "bpe.hh"
#include "tokenizer.hh"

namespace tokenizer {

namespace {

absl::flat_hash_map<std::string,
                    int>
CountChunks(const std::vector<std::filesystem::path>& paths,
            const boost::regex& pattern) {
    absl::flat_hash_map<std::string, int> counts;

    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) continue;

        auto size = file.tellg();
        file.seekg(0);
        std::string content(static_cast<size_t>(size), '\0');
        file.read(content.data(), size);
        content.resize(static_cast<size_t>(file.gcount()));

        boost::cregex_iterator it(content.data(), content.data() + content.size(), pattern);
        boost::cregex_iterator end;
        for (; it != end; ++it) {
            const auto& match = (*it)[0];
            std::string_view token(match.first, match.length());
            counts[token]++;
        }
    }

    return counts;
}

}  // namespace

void Tokenizer::Train(const std::vector<std::filesystem::path>& paths,
                      int vocab_size,
                      const boost::regex& pattern) {
    assert(vocab_size <= 65536 && "vocab_size exceeds 16-bit limit");

    auto global_counts = CountChunks(paths, pattern);

    std::vector<std::vector<uint32_t>> words;
    std::vector<int> word_counts;
    words.reserve(global_counts.size());
    word_counts.reserve(global_counts.size());

    for (const auto& [word, count] : global_counts) {
        std::vector<uint32_t> byte_ids;
        byte_ids.reserve(word.size());
        for (unsigned char c : word) byte_ids.push_back(c);
        words.push_back(std::move(byte_ids));
        word_counts.emplace_back(count);
    }

    merges_ = TrainBPE(words, word_counts, vocab_size - 256);
}

std::vector<Tokenizer::Token> Tokenizer::Encode(const std::string text) {
    std::vector<Tokenizer::Token> tokens;
    std::priority_queue<uint32_t> id_pairs;

    size_t codepoint_index = 0;
    while (codepoint_index < text.size()) {
        size_t len = 0;

        unsigned char first_byte = static_cast<unsigned char>(text[codepoint_index]);

        if ((first_byte & 0x80) == 0x00)
            len = 1;
        else if ((first_byte & 0xE0) == 0xC0)
            len = 2;
        else if ((first_byte & 0xF0) == 0xE0)
            len = 3;
        else if ((first_byte & 0xF8) == 0xF0)
            len = 4;

        Token token{};
        for (size_t i = 0; i < len; ++i) {
            token = (token << 8) | static_cast<unsigned char>(text[codepoint_index + i]);
        }

        tokens.push_back(token);
        codepoint_index += len;
    }
}

}  // namespace tokenizer
