import tempfile
import unittest
from pathlib import Path

from app import PaperStore


class PaperStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store_path = Path(self.tmp.name) / "papers.json"
        self.store = PaperStore(store_path=self.store_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_add_and_search_text(self) -> None:
        text = "딥러닝은 컴퓨터 비전의 중요한 기법이다.\n\nTransformer 기반 모델이 많이 사용된다."
        self.store.add_paper_from_text("vision", "local", text)

        hits = self.store.search("Transformer로 비전을 개선하는 방법?", top_k=1)

        self.assertTrue(hits, "검색 결과가 반환되어야 합니다.")
        self.assertIn("transformer", hits[0].chunk.lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
