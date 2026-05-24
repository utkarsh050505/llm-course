import os, json
import regex as re
from typing import IO, BinaryIO, Iterable, Optional, Type, Iterator
from collections import defaultdict
from . import helper_functions
import numpy.typing as npt
import torch
import heapq

class MergePriority:
    def __init__(self, freq, pair):
        self.freq = freq
        self.pair = pair
        
    def __lt__(self, other):
        if self.freq != other.freq:  # If no Tie
            # Normally heapq removes the smallest val first, but this reverses the order
            return self.freq > other.freq  # The highest frequency is treated as "smaller"
        return self.pair > other.pair  # If tie, lexicographically highest pair bubbles up

# Tokenizer
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if special_tokens:
            max_id = max(vocab.keys()) if vocab else -1
            for special_token in special_tokens:
                special_bytes = special_token.encode('utf-8')
                if special_bytes not in vocab.values():
                    max_id += 1
                    vocab[max_id] = special_bytes

        self.bytes_to_id = {v: k for k, v in vocab.items()}
    
    def encode(self, text: str) -> list[int]:
        spl_tokens = self.special_tokens if self.special_tokens else []
        spl_tokens_set = {
            t.encode('utf-8') if isinstance(t, str) else t
            for t in spl_tokens
        }

        # Get pre-token CHUNKS (not flat bytes)
        # We need to split into chunks first, then BPE each chunk independently
        if spl_tokens:
            sorted_special = sorted(spl_tokens, key=len, reverse=True)
            split_pattern = "|".join(re.escape(tok) for tok in sorted_special)
            split_regex = re.compile(f"({split_pattern})")
            parts = split_regex.split(text)
        else:
            parts = [text]

        word_pattern = helper_functions.build_pretokenizer_pattern([])  # no special tokens
        result_ids = []

        for part in parts:
            if not part:
                continue

            part_bytes = part.encode('utf-8') if isinstance(part, str) else part
            if part_bytes in spl_tokens_set:
                # Special token: emit as single token, no BPE
                result_ids.append(self.bytes_to_id[part_bytes])
                continue

            # Regular text: find each pre-token chunk and BPE independently
            for m in word_pattern.finditer(part):
                word = m.group(0)
                # Start with individual bytes for this word
                tokens = [bytes([b]) for b in word.encode('utf-8')]

                # Apply merges within this word only
                for pair in self.merges:
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                            new_tokens.append(tokens[i] + tokens[i+1])
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens

                for tok in tokens:
                    result_ids.append(self.bytes_to_id[tok])

        return result_ids
    
    def decode(self, ids: list[int]) -> str:
        tokens_bytes = [self.vocab[idx] for idx in ids]
        return b"".join(tokens_bytes).decode("utf-8", errors="replace")
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        spl_tokens = special_tokens if special_tokens else []

        with open(vocab_filepath, "r") as vocab_file:
            vocab = {}
            data = json.load(vocab_file)

            for i in data:
                vocab[data[i]] = i.encode("utf-8")
        
        with open(merges_filepath, "r", encoding='utf-8') as merge_file:
            merges = []
            
            for line in merge_file:
                line = line.rstrip("\r\n")
                if not line:
                    continue

                parts = line.split(" ")
                if len(parts) == 2:
                    pair = (parts[0].encode("utf-8"), parts[1].encode("utf-8"))
                    merges.append(pair)

        return cls(vocab, merges, spl_tokens)        
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        pattern = helper_functions.build_pretokenizer_pattern(self.special_tokens or [])
        
        for chunk in iterable:
            buffer += chunk
            matches = list(pattern.finditer(buffer))
            
            if len(matches) > 1:
                safe_boundary = matches[-2].end()
                safe_text = buffer[:safe_boundary]
                yield from self.encode(safe_text)
                buffer = buffer[safe_boundary:]
        
        if buffer:
            yield from self.encode(buffer)