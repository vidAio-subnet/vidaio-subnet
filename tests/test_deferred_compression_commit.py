import ast
from pathlib import Path
import unittest


class DeferredCompressionCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        validator_path = Path(__file__).parents[1] / "neurons" / "validator.py"
        validator_tree = ast.parse(validator_path.read_text(encoding="utf-8"))
        validator_class = next(
            node
            for node in validator_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Validator"
        )
        cls.methods = {
            node.name: node
            for node in validator_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_compression_batches_are_scored_before_the_round_is_committed(self):
        method = self.methods["process_compression_miners"]
        scoring_loop = next(
            node
            for node in method.body
            if isinstance(node, ast.For)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "score_compressions"
                for child in ast.walk(node)
            )
        )
        score_call = next(
            child
            for child in ast.walk(scoring_loop)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "score_compressions"
        )
        commit_keyword = next(
            keyword
            for keyword in score_call.keywords
            if keyword.arg == "commit_scores"
        )
        self.assertIsInstance(commit_keyword.value, ast.Constant)
        self.assertIs(commit_keyword.value.value, False)

        loop_position = method.body.index(scoring_loop)
        later_calls = [
            child
            for statement in method.body[loop_position + 1 :]
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
        ]
        self.assertTrue(
            any(call.func.attr == "_commit_compression_scores" for call in later_calls)
        )

    def test_score_compressions_preserves_immediate_commit_by_default(self):
        method = self.methods["score_compressions"]
        commit_parameter = method.args.args[-1]
        self.assertEqual(commit_parameter.arg, "commit_scores")
        self.assertIs(method.args.defaults[-1].value, True)

        conditional_commit = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "commit_scores"
        )
        self.assertTrue(
            any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "_commit_compression_scores"
                for child in ast.walk(conditional_commit)
            )
        )


if __name__ == "__main__":
    unittest.main()
