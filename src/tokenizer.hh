#ifndef NANOGPTCPP_TOKENIZER_H_
#define NANOGPTCPP_TOKENIZER_H_

#include <absl/container/flat_hash_map.h>

#include <boost/regex.hpp>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace tokenizer {

class Tokenizer {
    using Token = uint32_t;

   public:
    Tokenizer() = default;

    void Train(const std::vector<std::filesystem::path>& files,
               int vocab_size,
               const boost::regex& pattern);
    std::vector<Token> Encode(std::string text);

   private:
    absl::flat_hash_map<uint32_t, uint32_t> merges_;
    absl::flat_hash_map<Token, uint32_t> tokens_to_ids_;
    absl::flat_hash_map<uint32_t, Token> ids_to_tokens_;
};

}  // namespace tokenizer

#endif
