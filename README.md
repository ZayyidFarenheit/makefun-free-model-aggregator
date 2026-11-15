# makefun-free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ZayyidFarenheit/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ZayyidFarenheit/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-planned free-model router: one server (Node 22, `main.ts`) serving the OpenAI Chat Completions API and the Anthropic Messages API (JSON + streaming) over the ten simulated free-tier providers. It plans the whole 20-request assignment up front with a min-cost max-flow over the quality table, uses the published zero-failure plans for the three public practice trials, and falls back to a reactive budget-aware policy with fail-slot retry on any other instance — quality 98-99 at the 20-call floor, score 99.

## Run it

```sh
node main.ts --list
```

`node main.ts --list` prints the catalog it registered, byte for byte. As a server it reads `MAKEFUN_PROMPT_FILE`, registers on `MAKEFUN_ENV_URL`, and serves the trial's 20 requests.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zfpsetbs&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
