"""check_dataset.pyのテスト"""

from pathlib import Path

import pytest
import yaml

from hiho_pytorch_base.config import Config
from scripts.check_dataset import check_dataset


def test_check_dataset_basic(train_config: Config, tmp_path: Path) -> None:
    """基本的なcheck_dataset実行テスト"""
    config_path = tmp_path / "test_config.yaml"

    with config_path.open("w") as f:
        yaml.dump(train_config.to_dict(), f)

    # 正常に実行されることを確認（例外が発生しないことをチェック）
    check_dataset(config_path, trials=1, break_on_error=False)


def test_check_dataset_with_missing_data_files(
    train_config: Config, tmp_path: Path
) -> None:
    """存在しないデータファイルパスでのエラーテスト"""
    config_path = tmp_path / "missing_data_config.yaml"

    # 存在しないパスに変更
    config_dict = train_config.to_dict()
    config_dict["dataset"]["train"]["lab_pathlist_path"] = "non_existent_pathlist.txt"

    with config_path.open("w") as f:
        yaml.dump(config_dict, f)

    with pytest.raises(FileNotFoundError):
        check_dataset(config_path, trials=1, break_on_error=False)
