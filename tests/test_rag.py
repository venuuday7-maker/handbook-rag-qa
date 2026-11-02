import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag import Retriever, chunk_markdown  # noqa: E402

MD = "### 1.1 Cats\nCats purr and chase mice around the house.\n### 1.2 Rockets\nRockets burn fuel to reach orbit.\n"


def test_chunking_keeps_section_numbers():
    assert [(c.section, c.title) for c in chunk_markdown(MD)] == [("1.1", "Cats"), ("1.2", "Rockets")]


def test_long_section_is_windowed():
    long_md = "### 3.1 Long\n" + " ".join(["word"] * 500)
    assert len(chunk_markdown(long_md)) > 1


def test_retrieval_finds_right_section():
    r = Retriever(chunk_markdown(MD))
    assert r.ranked_sections("how do rockets get to orbit")[0] == "1.2"


def test_real_handbook_loads():
    assert len(Retriever.from_dir().chunks) > 20
