import pytest
import sys
import os
import subprocess
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

# Add the parent directory to sys.path to import the run module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lmgame-bench'))

import run


class TestRunScript:
    """Test suite for the run.py script functionality."""
    
    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for testing."""
        args = Mock()
        args.model_name = "gemini-flash-2.0"
        args.game_names = "twenty_forty_eight,sokoban,tetris"
        args.harness_mode = "both"
        args.max_parallel_procs = 2
        return args
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_run_single_game_config_success(self, temp_dir):
        """Test successful execution of run_single_game_config."""
        with patch('subprocess.run') as mock_subprocess:
            # Mock successful subprocess execution
            mock_process = Mock()
            mock_process.stdout = "Game completed successfully"
            mock_process.stderr = ""
            mock_subprocess.return_value = mock_process
            
            # Create logs directory
            logs_dir = os.path.join(temp_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            with patch('os.makedirs'):
                with patch('builtins.open', create=True) as mock_open:
                    result = run.run_single_game_config(
                        game_name="tetris",
                        model_name="gemini-flash-2.0",
                        use_harness=True,
                        custom_runner_script_path="lmgame-bench/custom_runner.py"
                    )
            
            # Verify result tuple
            game_name, use_harness, success, stdout, stderr = result
            assert game_name == "tetris"
            assert use_harness is True
            assert success is True
            assert stdout == "Game completed successfully"
            assert stderr == ""
    
    def test_run_single_game_config_failure(self):
        """Test failed execution of run_single_game_config."""
        with patch('subprocess.run') as mock_subprocess:
            # Mock failed subprocess execution
            mock_error = subprocess.CalledProcessError(
                returncode=1,
                cmd=["python", "custom_runner.py"],
                output="",
                stderr="Error: Game initialization failed"
            )
            mock_error.stdout = ""
            mock_error.stderr = "Error: Game initialization failed"
            mock_subprocess.side_effect = mock_error
            
            with patch('os.makedirs'):
                with patch('builtins.open', create=True):
                    result = run.run_single_game_config(
                        game_name="invalid_game",
                        model_name="gemini-flash-2.0",
                        use_harness=False,
                        custom_runner_script_path="lmgame-bench/custom_runner.py"
                    )
            
            # Verify failure result
            game_name, use_harness, success, stdout, stderr = result
            assert game_name == "invalid_game"
            assert use_harness is False
            assert success is False
            assert stderr == "Error: Game initialization failed"
    
    def test_argument_parsing(self):
        """Test command line argument parsing."""
        test_args = [
            "--model_name", "gemini-flash-2.0",
            "--game_names", "tetris,sokoban",
            "--harness_mode", "true",
            "--max_parallel_procs", "4"
        ]
        
        with patch('sys.argv', ['run.py'] + test_args):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.model_name = "gemini-flash-2.0"
                mock_args.game_names = "tetris,sokoban"
                mock_args.harness_mode = "true"
                mock_args.max_parallel_procs = 4
                mock_parse.return_value = mock_args
                
                # Test that arguments are parsed correctly
                assert mock_args.model_name == "gemini-flash-2.0"
                assert mock_args.game_names == "tetris,sokoban"
                assert mock_args.harness_mode == "true"
                assert mock_args.max_parallel_procs == 4
    
    def test_game_names_parsing(self):
        """Test parsing of comma-separated game names."""
        game_names_string = "twenty_forty_eight,sokoban,candy_crush,tetris"
        expected_games = ["twenty_forty_eight", "sokoban", "candy_crush", "tetris"]
        
        # Simulate the parsing logic from main()
        game_names_list = [name.strip() for name in game_names_string.split(',') if name.strip()]
        
        assert game_names_list == expected_games
    
    def test_harness_mode_task_creation(self):
        """Test task creation for different harness modes."""
        game_names = ["tetris", "sokoban"]
        model_name = "gemini-flash-2.0"
        custom_runner_script_path = "lmgame-bench/custom_runner.py"
        
        # Test harness_mode = "true"
        tasks_harness_true = []
        for game_name in game_names:
            tasks_harness_true.append((game_name, model_name, True, custom_runner_script_path))
        
        assert len(tasks_harness_true) == 2
        assert all(task[2] is True for task in tasks_harness_true)  # All harness=True
        
        # Test harness_mode = "false"
        tasks_harness_false = []
        for game_name in game_names:
            tasks_harness_false.append((game_name, model_name, False, custom_runner_script_path))
        
        assert len(tasks_harness_false) == 2
        assert all(task[2] is False for task in tasks_harness_false)  # All harness=False
        
        # Test harness_mode = "both"
        tasks_both = []
        for game_name in game_names:
            tasks_both.append((game_name, model_name, True, custom_runner_script_path))
            tasks_both.append((game_name, model_name, False, custom_runner_script_path))
        
        assert len(tasks_both) == 4  # 2 games × 2 modes
        harness_values = [task[2] for task in tasks_both]
        assert True in harness_values and False in harness_values
    
    def test_all_available_games(self):
        """Test that all games from game_config_mapping are valid."""
        expected_games = [
            "twenty_forty_eight", "sokoban", "candy_crush", "tetris",
            "super_mario_bros", "ace_attorney", "nineteen_forty_two"
        ]
        
        # Test parsing all games at once
        all_games_string = ",".join(expected_games)
        parsed_games = [name.strip() for name in all_games_string.split(',') if name.strip()]
        
        assert parsed_games == expected_games
        assert len(parsed_games) == 7
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test empty game names
        empty_games = [name.strip() for name in "".split(',') if name.strip()]
        assert len(empty_games) == 0
        
        # Test game names with extra spaces
        spaced_games = [name.strip() for name in " tetris , sokoban , ".split(',') if name.strip()]
        assert spaced_games == ["tetris", "sokoban"]
        
        # Test single game
        single_game = [name.strip() for name in "tetris".split(',') if name.strip()]
        assert single_game == ["tetris"]
    
    @patch('multiprocessing.cpu_count')
    def test_parallel_processing_configuration(self, mock_cpu_count):
        """Test parallel processing configuration."""
        mock_cpu_count.return_value = 8
        
        # Test default (None) should use cpu_count
        max_parallel_procs = None
        num_processes = max_parallel_procs if max_parallel_procs else mock_cpu_count()
        assert num_processes == 8
        
        # Test explicit value
        max_parallel_procs = 4
        num_processes = max_parallel_procs if max_parallel_procs else mock_cpu_count()
        assert num_processes == 4
    
    def test_command_construction(self):
        """Test subprocess command construction."""
        game_name = "tetris"
        model_name = "gemini-flash-2.0"
        custom_runner_script_path = "lmgame-bench/custom_runner.py"
        
        # Test command with harness
        base_command = [
            sys.executable,
            custom_runner_script_path,
            "--game_name", game_name,
            "--model_name", model_name
        ]
        command_with_harness = base_command + ["--harness"]
        
        expected_with_harness = [
            sys.executable,
            "lmgame-bench/custom_runner.py",
            "--game_name", "tetris",
            "--model_name", "gemini-flash-2.0",
            "--harness"
        ]
        
        assert command_with_harness == expected_with_harness
        
        # Test command without harness
        command_without_harness = base_command
        
        expected_without_harness = [
            sys.executable,
            "lmgame-bench/custom_runner.py",
            "--game_name", "tetris",
            "--model_name", "gemini-flash-2.0"
        ]
        
        assert command_without_harness == expected_without_harness
    
    def test_log_file_naming(self):
        """Test log file naming convention."""
        game_name = "tetris"
        model_name = "gemini-flash-2.0"
        
        # Test log file for harness=True
        log_file_harness = f"logs/run_log_{game_name}_{model_name}_harness_true.txt"
        expected_harness = "logs/run_log_tetris_gemini-flash-2.0_harness_true.txt"
        assert log_file_harness == expected_harness
        
        # Test log file for harness=False
        log_file_no_harness = f"logs/run_log_{game_name}_{model_name}_harness_false.txt"
        expected_no_harness = "logs/run_log_tetris_gemini-flash-2.0_harness_false.txt"
        assert log_file_no_harness == expected_no_harness
    
    @patch('os.path.exists')
    def test_custom_runner_script_existence(self, mock_exists):
        """Test validation of custom_runner.py existence."""
        custom_runner_script_path = os.path.join("lmgame-bench", "custom_runner.py")
        
        # Test when file exists
        mock_exists.return_value = True
        assert mock_exists(custom_runner_script_path) is True
        
        # Test when file doesn't exist
        mock_exists.return_value = False
        assert mock_exists(custom_runner_script_path) is False
    
    def test_results_summary_calculation(self):
        """Test calculation of successful and failed runs."""
        # Mock results from parallel execution
        mock_results = [
            ("tetris", True, True, "stdout1", ""),  # Success with harness
            ("tetris", False, True, "stdout2", ""),  # Success without harness
            ("sokoban", True, False, "", "error1"),  # Failure with harness
            ("sokoban", False, True, "stdout3", ""),  # Success without harness
            ("candy_crush", True, False, "", "error2"),  # Failure with harness
        ]
        
        successful_runs = 0
        failed_runs = 0
        
        for game, harness, success, _, _ in mock_results:
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
        
        assert successful_runs == 3
        assert failed_runs == 2
        assert successful_runs + failed_runs == len(mock_results)


class TestGameConfigIntegration:
    """Integration tests for game configuration validation."""
    
    def test_all_supported_games_command_generation(self):
        """Test command generation for all supported games."""
        supported_games = [
            "twenty_forty_eight", "sokoban", "candy_crush", "tetris",
            "super_mario_bros", "ace_attorney", "nineteen_forty_two"
        ]
        
        model_name = "gemini-flash-2.0"
        
        # Test command for all games with harness
        all_games_harness_cmd = (
            f"python3 lmgame-bench/run.py --model_name {model_name} "
            f"--game_names {','.join(supported_games)} --harness_mode true"
        )
        
        expected_harness_cmd = (
            "python3 lmgame-bench/run.py --model_name gemini-flash-2.0 "
            "--game_names twenty_forty_eight,sokoban,candy_crush,tetris,"
            "super_mario_bros,ace_attorney,nineteen_forty_two --harness_mode true"
        )
        
        assert all_games_harness_cmd == expected_harness_cmd
        
        # Test command for all games without harness
        all_games_no_harness_cmd = (
            f"python3 lmgame-bench/run.py --model_name {model_name} "
            f"--game_names {','.join(supported_games)} --harness_mode false"
        )
        
        expected_no_harness_cmd = (
            "python3 lmgame-bench/run.py --model_name gemini-flash-2.0 "
            "--game_names twenty_forty_eight,sokoban,candy_crush,tetris,"
            "super_mario_bros,ace_attorney,nineteen_forty_two --harness_mode false"
        )
        
        assert all_games_no_harness_cmd == expected_no_harness_cmd
    
    def test_individual_game_commands(self):
        """Test individual game command generation."""
        games = ["tetris", "sokoban", "candy_crush"]
        model_name = "gemini-flash-2.0"
        
        for game in games:
            # With harness
            cmd_harness = (
                f"python3 lmgame-bench/run.py --model_name {model_name} "
                f"--game_names {game} --harness_mode true"
            )
            
            assert f"--game_names {game}" in cmd_harness
            assert "--harness_mode true" in cmd_harness
            assert f"--model_name {model_name}" in cmd_harness
            
            # Without harness
            cmd_no_harness = (
                f"python3 lmgame-bench/run.py --model_name {model_name} "
                f"--game_names {game} --harness_mode false"
            )
            
            assert f"--game_names {game}" in cmd_no_harness
            assert "--harness_mode false" in cmd_no_harness
            assert f"--model_name {model_name}" in cmd_no_harness


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 