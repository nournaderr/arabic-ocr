import pickle
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DawgNode:
    children: dict = field(default_factory=dict)
    is_end:   bool = False


def build_dawg(word_list: list[str]) -> DawgNode:
    """Insert all Arabic words into a trie and return the root node.

    Word list source: Arabic Gigaword corpus or ar.wiktionary.
    """
    root = DawgNode()
    for word in word_list:
        node = root
        for ch in word.strip():
            if ch not in node.children:
                node.children[ch] = DawgNode()
            node = node.children[ch]
        node.is_end = True
    return root


def dawg_search(root: DawgNode, prefix: str) -> list[str]:
    """Return all words in the trie that start with prefix."""
    node = root
    for ch in prefix:
        if ch not in node.children:
            return []
        node = node.children[ch]
    completions: list[str] = []
    _collect(node, prefix, completions)
    return completions


def _collect(node: DawgNode, current: str, results: list[str]) -> None:
    if node.is_end:
        results.append(current)
    for ch, child in node.children.items():
        _collect(child, current + ch, results)


def save_dawg(root: DawgNode, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(root, f)


def load_dawg(path: Path) -> DawgNode:
    with open(path, "rb") as f:
        return pickle.load(f)
