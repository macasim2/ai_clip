from src.launcher import model_installed


def test_model_installed_exact_and_latest():
    installed = ["llama3.2:latest", "llava:latest"]
    assert model_installed(installed, "llama3.2:latest")
    assert model_installed(installed, "llama3.2")
    assert model_installed(installed, "llava")
    assert not model_installed(installed, "qwen2.5:7b-instruct")
    assert model_installed(installed, "")
