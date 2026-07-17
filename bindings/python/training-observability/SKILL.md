---
name: training-observability
description: Inspect local or mounted machine-learning training runs through durable scalar metrics, rendered PNG plots, and saved visual predictions. Use when Codex needs to assess a live or completed run, compare metric trends, inspect prediction overlays, or diagnose training quality without modifying the run or relying on a live Visdom dashboard.
---

# Training Observability

Use `training-observe` against only the run root the user has placed in scope.
The command reads the run; `render` writes solely to its explicit output path.

## Inspect a run

1. Run `training-observe summary RUN` to establish status, latest scalar
   values, plot IDs, visual-artifact availability, and reader warnings.
2. Run `training-observe metrics RUN --name METRIC` for the raw points that
   support any claim about a trend. Treat elapsed-time and progress fields as
   operational signals, not model-quality metrics.
3. Run `training-observe plots RUN`, then render only relevant plots, for
   example:

   ```shell
   training-observe render RUN --plot loss-train-loss --output /tmp/train-loss.png
   ```

   Inspect the emitted PNG with the available image-viewing capability.
4. Run `training-observe artifacts RUN --step latest` and inspect both each
   returned image and its paired JSON metadata before judging prediction quality.

## Report evidence

- State the exact metric names, step range, and values that support a trend.
- Separate observations from inferences. Report a loss plateau before inferring
  that an optimizer or data change could help.
- Compare per-test-set traces independently; do not average them by eye.
- Treat missing plots or results as missing evidence, not a good or bad result.

## Preserve control boundaries

Do not start, stop, cancel, resume, delete, or reconfigure a training run.
Do not scrape the live Visdom UI as the source of truth. Use the flushed metric
stream and saved visual artifacts, which remain available while the run is live
and after the Visdom server exits.
