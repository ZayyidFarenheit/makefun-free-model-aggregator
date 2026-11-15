# Free Model Aggregator — make.fun entry

Quality-planned free-model router for `free-model-aggregator-01`.

One server (`main.ts`, Node 22) that serves the OpenAI Chat Completions API
and the Anthropic Messages API (JSON + streaming) over the ten simulated
free-tier providers the trial gives it. It reads the brief, plans the whole
20-request assignment up front from the quality table, each provider's budget
and failure schedule, then serves every request with the best free model that
can serve it — dropping a spent provider on its first refusal and never
wasting a provider call.

`node main.ts --list` prints the catalog it registered, byte for byte.
