#include <cassert>
#include <format>
#include <fstream>
#include <queue>
#include <string>

#include <absl/container/flat_hash_map.h>
#include <boost/regex.hpp>
#include <spdlog/spdlog.h>

#include "bpe.hh"
#include "tokenizer.hh"

namespace tokenizer
{

    // split texts into semantic chunks and maintain counts of each chunk
    absl::flat_hash_map<std::string, int>
    SplitAndCountChunks(const std::vector<std::filesystem::path>& paths, const boost::regex& pattern)
    {
        absl::flat_hash_map<std::string, int> counts;

        for (const auto& path : paths)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open() || file.fail())
            {
                throw std::runtime_error(std::format("Failed to open file {}", path.string()));
            }

            auto size = file.tellg();
            file.seekg(0);

            std::string content(static_cast<size_t>(size), '\0');
            file.read(content.data(), size);

            auto matches_begin = boost::sregex_iterator(content.begin(), content.end(), pattern);
            auto matches_end = boost::sregex_iterator();
            for (boost::sregex_iterator i = matches_begin; i != matches_end; ++i)
            {
                counts[i->str()]++;
            }
        }

        return counts;
    }

    void Tokenizer::Save(const std::filesystem::path& path) const
    {
        std::ofstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for saving tokenizer: " + path.string());
        }
        for (const auto& [pair, vocab_id] : merges_)
        {
            file << vocab_id << " " << pair.first << " " << pair.second << "\n";
        }

        spdlog::info("Merges saved to {}", path.string());
    }

    void
    Tokenizer::Train(const std::vector<std::filesystem::path>& paths, int vocab_size, const boost::regex& pattern)
    {
        assert(vocab_size <= 65536 && "vocab_size exceeds 16-bit limit");

        spdlog::info("started chunking");
        auto chunks_to_counts = SplitAndCountChunks(paths, pattern);
        spdlog::info("finished chunk: {} chunks", chunks_to_counts.size());

        std::vector<std::vector<Token>> chunks;
        std::vector<int> chunk_counts;
        chunks.reserve(chunks_to_counts.size());
        chunk_counts.reserve(chunks_to_counts.size());

        for (const auto& [word, count] : chunks_to_counts)
        {
            std::vector<uint32_t> byte_ids;
            byte_ids.reserve(word.size());
            for (unsigned char c : word)
            {
                byte_ids.push_back(c);
            }
            chunks.push_back(std::move(byte_ids));
            chunk_counts.emplace_back(count);
        }

        merges_ = TrainBPE(chunks, chunk_counts, vocab_size - 256);
    }

    std::vector<Token> Tokenizer::Encode(const std::string& text)
    {
        std::vector<Token> tokens;
        std::priority_queue<Token> id_pairs;

        size_t codepoint_index = 0;
        while (codepoint_index < text.size())
        {
            size_t len{};

            unsigned char first_byte = static_cast<unsigned char>(text[codepoint_index]);

            if ((first_byte & 0x80) == 0x00)
            {
                len = 1;
            }
            else if ((first_byte & 0xE0) == 0xC0)
            {
                len = 2;
            }
            else if ((first_byte & 0xF0) == 0xE0)
            {
                len = 3;
            }
            else if ((first_byte & 0xF8) == 0xF0)
            {
                len = 4;
            }

            Token token{};
            for (size_t i = 0; i < len; ++i)
            {
                token = (token << 8) | static_cast<unsigned char>(text[codepoint_index + i]);
            }

            tokens.push_back(token);
            codepoint_index += len;
        }

        return tokens;
    }

}  // namespace tokenizer
