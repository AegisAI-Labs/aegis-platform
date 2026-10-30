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

| Scenario | Expected | Actual | Pass/Fail | Notes |
|---|---|---|---|---|
| Immediate success | 1 attempt | _fill in_ | _fill in_ | |
| Transient failure then success | Recover within budget | _fill in_ | _fill in_ | |
| All attempts fail | Stop at configured max | _fill in_ | _fill in_ | |
| Permanent failure | No retry | _fill in_ | _fill in_ | |
| Deadline | No attempt after deadline | _fill in_ | _fill in_ | |
| Stream fails before first chunk | Retry if budget remains | _fill in_ | _fill in_ | |
| Stream fails after first chunk | Propagate; no restart | _fill in_ | _fill in_ | |
| Cancellation | Propagate; no retry | _fill in_ | _fill in_ | |

Record attempt counts and elapsed times. Mocked measurements do not establish real-world provider reliability.

## Conclusion
_Fill in after running the experiment, including deviations and follow-up work._
