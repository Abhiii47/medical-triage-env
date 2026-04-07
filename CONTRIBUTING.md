# Contributing to Medical Triage OpenEnv

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/MyFeature`)
3. Commit your changes (`git commit -am 'Add MyFeature'`)
4. Push to the branch (`git push origin feature/MyFeature`)
5. Open a Pull Request

## Adding Scenarios
To add a scenario, edit `tasks.py` and modify `SCENARIOS`. Ensure you also update `EXPECTED` patterns in `grader.py`.

## Adding Tests
Run tests with `python tests/test_env.py` and ensure they pass before submitting PRs.
