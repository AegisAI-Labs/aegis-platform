# Day 7 Experiment — Provider Retry and Failure Behavior

- Status: Planned; fill results only after execution.
- Run date: _fill in_
- Commit: _fill in_
- Python/OS/dependencies: _fill in_
- Providers/models: _fill in actual values_

## Questions
1. Does a transient failure recover within the attempt budget?
2. Are permanent failures returned without retry?
3. Does the overall deadline bound elapsed time?
4. Does streaming avoid restarting after its first chunk?
5. Are logs useful and free of sensitive data?

## Results

(aegis-platform) PS C:\Users\kader\pers\learn\Aegis\aegis-platform> uv run python -m scripts.experiment_retry_reliability
Retry reliability experiment (scripted provider; no network)
Iterations per scenario: 10
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry

Generate retries a transient timeout:
  recovery: iterations=10, mean=0.015075s, max=0.016265s
  attempts=20; expected=20
  result=PASS
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed
llm_retry
llm_retry
llm_request_failed

Generate stops at max_attempts when failures persist:
  attempts=30; expected=30
  result=PASS
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry
llm_retry

Stream retries a timeout before the first chunk:
  recovery: iterations=10, mean=0.015507s, max=0.016020s
  attempts=20; expected=20
  result=PASS

Stream does not retry after delivering partial output:
  attempts=10; expected=10
  result=PASS

Overall: PASS

## Conclusion
_Fill in after running the experiment, including deviations and follow-up work._
