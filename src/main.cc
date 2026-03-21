#include <iostream>
#include <string>

#include <boost/regex.hpp>

#include "tokenizer.hh"

int main()
{
    tokenizer::Tokenizer t;

    const boost::regex pattern(
        R"('(?i:[sdmt]|ll|ve|re)|[^\r\na-zA-Z0-9]?+[a-zA-Z]+|[0-9]{1,3}| ?[^\sa-zA-Z0-9]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)",
        boost::regex::perl | boost::regex::optimize
    );

    t.Train({"data/tinystories_train.txt"}, 50257, pattern);
    t.Save({"merges.txt"});

    while (true)
    {
        std::string prompt;
        std::cout << "Type something: ";
        std::getline(std::cin, prompt);
    }

    return 0;
}
