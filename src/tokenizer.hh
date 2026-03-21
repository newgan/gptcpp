#ifndef GPTCPP_TOKENIZER_H_
#define GPTCPP_TOKENIZER_H_

#include <cstdint>
#include <filesystem>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <boost/regex.hpp>

namespace tokenizer
{
    using Token = uint32_t;
    using TokenPair = std::pair<Token, Token>;

    class Tokenizer
    {
    private:


    public:

        Tokenizer() = default;

        void
        Train(const std::vector<std::filesystem::path>& paths, int vocab_size, const boost::regex& pattern);
        void Save(const std::filesystem::path& path) const;

        std::vector<Token> Encode(const std::string& text);

    private:

        absl::flat_hash_map<TokenPair, uint32_t> merges_;
        absl::flat_hash_map<Token, uint32_t> tokens_to_ids_;
        absl::flat_hash_map<uint32_t, Token> ids_to_tokens_;
    };

}  // namespace tokenizer

#endif
