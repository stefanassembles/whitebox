#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <memory>

using json = nlohmann::json;
namespace py = pybind11;

class TrieNode {
public:
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    bool is_end_of_word = false;
    std::string token;
    int token_id = -1;
};

class WordPieceTrie {
private:
    std::unique_ptr<TrieNode> root;
    std::string unk_token;
    int unk_token_id;
    std::string prefix = "##";

public:
    WordPieceTrie(const std::string& unk = "[UNK]", int unk_id = 0)
        : root(std::make_unique<TrieNode>()), unk_token(unk), unk_token_id(unk_id) {}

    void insert(const std::string& token, int token_id) {
        if (token_id < 0) {
            throw std::invalid_argument("❌ Cannot insert token with ID < 0: " + token);
        }

        TrieNode* node = root.get();
        for (char c : token) {
            if (!node->children[c]) {
                node->children[c] = std::make_unique<TrieNode>();
            }
            node = node->children[c].get();
        }

        node->is_end_of_word = true;
        node->token = token;
        node->token_id = token_id;
    }

    std::pair<std::string, int> find_longest_token(const std::string& text, size_t start) {
        TrieNode* node = root.get();
        std::string longest_match = "";
        int longest_match_id = -1;

        for (size_t i = start; i < text.length(); ++i) {
            char c = text[i];
            if (!node->children[c]) {
                break;
            }
            node = node->children[c].get();
            if (node->is_end_of_word) {
                longest_match = node->token;
                longest_match_id = node->token_id;
            }
        }

        return {longest_match, longest_match_id};
    }

    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> result;
        size_t start = 0;

        while (start < text.length()) {
            auto [token, token_id] = find_longest_token(text, start);

            if (!token.empty()) {
                result.push_back(token_id);
                start += token.length();
            } else {
                result.push_back(unk_token_id);
                start++;
            }
        }

        return result;
    }

    std::vector<std::string> ids_to_tokens(const std::vector<int>& ids) {
        std::unordered_map<int, std::string> id_to_token_map;
        build_id_to_token_map(root.get(), "", id_to_token_map);

        std::vector<std::string> tokens;
        for (int id : ids) {
            std::cout << "[TRACE] Looking up ID: " << id << "\n";
            if (id_to_token_map.count(id)) {
                tokens.push_back(id_to_token_map[id]);
            } else {
                std::cerr << "⚠️  Unknown token ID: " << id << " → using [UNK]\n";
                tokens.push_back(unk_token);
            }
        }
        return tokens;
    }

private:
    void build_id_to_token_map(TrieNode* node, std::string prefix, std::unordered_map<int, std::string>& map) {
        if (!node) {
            std::cerr << "❌ Null TrieNode encountered\n";
            return;
        }

        if (node->is_end_of_word) {
            if (node->token_id < 0) {
                std::cerr << "❌ ERROR: Node '" << node->token << "' has invalid token_id -1\n";
                return;
            }

            if (map.count(node->token_id)) {
                std::cerr << "⚠️ Duplicate token ID: " << node->token_id
                          << " for token '" << node->token << "'\n";
            }

            map[node->token_id] = node->token;
        }

        for (auto& [c, child] : node->children) {
            build_id_to_token_map(child.get(), prefix + c, map);
        }
    }
};

std::unordered_map<std::string, int> extract_vocab_from_tokenizer_json(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + json_path);
    }

    json tokenizer_data;
    file >> tokenizer_data;

    std::unordered_map<std::string, int> vocab;

    if (tokenizer_data.contains("model") &&
        tokenizer_data["model"].contains("vocab")) {
        auto vocab_obj = tokenizer_data["model"]["vocab"];
        for (auto it = vocab_obj.begin(); it != vocab_obj.end(); ++it) {
            vocab[it.key()] = it.value();
        }
    } else if (tokenizer_data.contains("model") &&
               tokenizer_data["model"].contains("pieces")) {
        auto pieces = tokenizer_data["model"]["pieces"];
        for (size_t i = 0; i < pieces.size(); i++) {
            vocab[pieces[i]["piece"]] = pieces[i]["id"];
        }
    }

    return vocab;
}

void load_vocab_into_trie(WordPieceTrie& trie, const std::unordered_map<std::string, int>& vocab) {
    for (const auto& [token, id] : vocab) {
        if (id < 0) {
            std::cerr << "❌ Invalid ID " << id << " for token '" << token << "'\n";
        }
        trie.insert(token, id);
    }
}

class Tokenizer {
private:
    WordPieceTrie trie;

public:
    Tokenizer(const std::string& json_path) {
        std::cout << "[TRACE] Loading vocab from: " << json_path << std::endl;
        auto vocab = extract_vocab_from_tokenizer_json(json_path);
        std::cout << "[TRACE] Loaded vocab of size: " << vocab.size() << std::endl;
        if (!vocab.count("[UNK]")) {
            std::cerr << "❌ [UNK] token not found in vocab! Consider adding it.\n";
        }
        load_vocab_into_trie(trie, vocab);
    }

    std::vector<int> tokenize(const std::string& text) {
        std::cout << "[TRACE] Tokenizing input: " << text << "\n";
        auto ids = trie.tokenize(text);
        std::cout << "[TRACE] Token IDs: ";
        for (auto id : ids) std::cout << id << " ";
        std::cout << "\n";
        return ids;
    }

    std::vector<std::string> tokenize_to_strings(const std::string& text) {
        std::cout << "[TRACE] Tokenize to strings: " << text << "\n";
        auto ids = trie.tokenize(text);
        return trie.ids_to_tokens(ids);
    }
};

PYBIND11_MODULE(simple_tokenizer, m) {
    m.doc() = "WordPiece tokenizer implementation";

    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<const std::string&>(), "Load tokenizer from JSON vocab")
        .def("tokenize", &Tokenizer::tokenize, "Tokenize input string to token IDs")
        .def("tokenize_to_strings", &Tokenizer::tokenize_to_strings, "Tokenize input string to token strings (for debugging)");
}

