import sys

from pipeline.run_pipeline import build_arg_parser, main


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args([])
    assert args.max_papers == 150
    assert args.categories == ["cs.LG", "cs.AI", "cs.CL"]


def test_main_accepts_argv_and_ignores_sys_argv(monkeypatch):
    """Regression: Airflow's PythonOperator runs in a worker whose sys.argv is
    airflow's own command line. main() must NOT read sys.argv (it used to call
    parse_args() with no args -> SystemExit on the worker's argv)."""
    monkeypatch.setattr(sys, "argv", ["airflow", "tasks", "run", "arxiv_ingest_weekly"])
    # parsing an explicit empty argv must succeed regardless of sys.argv
    args = build_arg_parser().parse_args([])
    assert args.max_papers == 150
    # main must accept an argv parameter (the DAG passes [])
    assert "argv" in main.__code__.co_varnames
