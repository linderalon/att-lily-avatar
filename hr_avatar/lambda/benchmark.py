#!/usr/bin/env python3
"""
Benchmark: Iterative Analysis Pipeline (4 problems + synthesis)

Simulates the real browser flow: 4 parallel per-problem calls followed by
1 synthesis call.  Repeats N times and reports timing stats.

Usage:
    python3 benchmark.py                  # 10 iterations, default API URL
    python3 benchmark.py --runs 5         # 5 iterations
    python3 benchmark.py --url <url>      # custom API Gateway URL
    python3 benchmark.py --threshold 12   # custom pass/fail threshold (seconds)
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "https://30vsmo8j0l.execute-api.us-west-2.amazonaws.com/"
DEFAULT_RUNS = 10
DEFAULT_THRESHOLD_S = 19.0

# Full 4-problem transcript extracted from the real HAR capture.
TRANSCRIPT = [
    {"role": "assistant", "content": "Hey, I'm an AI Peer Programming Specialist. How are you doing today?"},
    {"role": "user", "content": "Awesome, what's up?"},
    {"role": "assistant", "content": "I'm doing great, thanks! I'm ready to help you with your coding challenge. We'll be working on a problem called \"Two Sum\" today. Take your time to read the problem description, and let me know when you're ready to start coding."},
    {"role": "user", "content": "Done."},
    {"role": "assistant", "content": "Nice work—all tests passing on Two Sum! Can you walk me through how your code handles the case where the `target` is found using the very first number in the `nums` array?"},
    {"role": "user", "content": "Because the problem is saying you may only use the same. There's no usage of the same element twice, this is never going to be a problem."},
    {"role": "assistant", "content": "That's a good point about the problem constraints. How does your code ensure that it doesn't use the same element twice, even if the complement happens to be the number itself?"},
    {"role": "user", "content": "Because I have a dictionary."},
    {"role": "assistant", "content": "You're right, the `seen` dictionary is key here. What is the time complexity of this solution, and what part of your code primarily determines it?"},
    {"role": "user", "content": "It's O of n and it's because we looped through the entire array."},
    {"role": "assistant", "content": "That's correct; the `for` loop makes it O(n) time complexity. Given that you're storing elements in the `seen` dictionary, what would you say is the space complexity of your solution?"},
    {"role": "user", "content": "Who have been?"},
    {"role": "assistant", "content": "The space complexity refers to the amount of memory your solution uses. In your code, you're using the `seen` dictionary to store elements. How does the size of this dictionary relate to the input `nums` array, and what does that imply for the space complexity?"},
    {"role": "user", "content": "It's O-O-N."},
    {"role": "assistant", "content": "That's right, it's O(n) for space complexity. Great job on Two Sum! That was problem one of four. Switching to the next challenge now."},
    {"role": "user", "content": "Done."},
    {"role": "assistant", "content": "Alright, here's Valid Palindrome. Take your time."},
    {"role": "user", "content": "Done done."},
    {"role": "assistant", "content": "Nice work—all tests passing on Valid Palindrome! Can you explain the purpose of your two `if` conditions inside the `while` loop, specifically why `continue` is used in those cases?"},
    {"role": "user", "content": "Sure, because I'm basically testing if the - Sure, because I'm basically testing if the... if one of the pointers is either right or left is If one of the pointers is either right or left is Um, not an alphanumeric And if it is, we're just skipping."},
    {"role": "assistant", "content": "That's a clear explanation of their function. What is the time complexity of this solution, and what operation within your code dictates it?"},
    {"role": "user", "content": "It's O of N because of the loop going through the entire string."},
    {"role": "assistant", "content": "That's correct, the single pass through the string makes it O(n) time complexity. What about the space complexity for this approach, Zohar?"},
    {"role": "user", "content": "It's O of 1 because we don't store anything. We just have two pointers."},
    {"role": "assistant", "content": "That's right, it's O(one) space complexity because you're only using a few variables for the pointers, regardless of the input string's size. Great job on Valid Palindrome! That was problem two of four. Switching to the next challenge now."},
    {"role": "user", "content": "Time completed."},
    {"role": "assistant", "content": "Alright, here's Reverse Linked List. Take your time."},
    {"role": "user", "content": "I finished."},
    {"role": "assistant", "content": "Nice work—all tests passing on Reverse Linked List! Can you explain how the `prev` and `curr` pointers are updated within the loop to correctly reverse the links?"},
    {"role": "user", "content": "I basically have three pointers, one, two, current, previous and next, and then in every duration I switch between next and previous."},
    {"role": "assistant", "content": "That's a good high-level overview. Could you elaborate on why it's important to store `curr.next` in a `next_node` variable before changing `curr.next` to `prev`?"},
    {"role": "user", "content": "because otherwise they'll confuse the pointers."},
    {"role": "assistant", "content": "That's right, it prevents losing the reference to the rest of the list. What is the time complexity of this solution, and what part of your code determines it?"},
    {"role": "user", "content": "O of n because we're in a loop looping through the entire list."},
    {"role": "assistant", "content": "That's correct, iterating through the list once makes it O(n) time complexity. What about the space complexity for this solution, Zohar?"},
    {"role": "user", "content": "2 of 1 because I only have 3 pointers."},
    {"role": "assistant", "content": "That's right, it's O(one) space complexity because you're only using a constant number of pointers regardless of the list's size. Great job on Reverse Linked List! That was problem three of four. Switching to the next challenge now."},
    {"role": "user", "content": "Finished."},
    {"role": "assistant", "content": "Alright, here's Fizz Buzz. Take your time."},
    {"role": "user", "content": "Finished."},
    {"role": "assistant", "content": "Nice work—all tests passing on Fizz Buzz! Can you explain how your bit manipulation approach with the `mask` variable correctly determines \"Fizz\", \"Buzz\", or \"FizzBuzz\" for each number?"},
    {"role": "user", "content": "Yeah, I basically have a mask of fizzbuzz. and then I loop through the string And then I'm basically testing each part If it's divisible by three or divisible by five, and if yes, then... I opened it and then uh That's the last final point. That's it."},
    {"role": "assistant", "content": "You're using a `mask` to encode the FizzBuzz pattern. How does the `code` variable, derived from shifting and masking, map to the `words` array to get the correct output string?"},
    {"role": "user", "content": "I'm not sure I understood the question."},
    {"role": "assistant", "content": "Let me rephrase. After you extract the `code` using `(mask >> (two * ((i - one) % fifteen))) & three`, how does that specific `code` value (zero, one, two, or three) tell your program whether to append \"Fizz\", \"Buzz\", \"FizzBuzz\", or the number itself?"},
    {"role": "user", "content": "based on the words array."},
    {"role": "assistant", "content": "Indeed, the `words` array is crucial for that mapping. What is the time complexity of this solution, and which part of your code primarily determines it?"},
    {"role": "user", "content": "O of n because of the for loop."},
    {"role": "assistant", "content": "That's correct, the `for` loop iterating up to `n` makes it O(n) time complexity. What would you say is the space complexity of your solution, Zohar?"},
    {"role": "user", "content": "One."},
    {"role": "assistant", "content": "Could you elaborate on why you believe the space complexity is O(one), considering you're building an `answer` list that grows with `n`?"},
    {"role": "user", "content": "No, actually it's O of N because I have an array, answer array."},
    {"role": "assistant", "content": "That's right, the `answer` array growing with `n` makes the space complexity O(n). Excellent! You've completed all four problems. Great work today! Ending the session now."},
]

PROBLEMS = [
    {"id": "two-sum",            "title": "Two Sum",            "difficulty": "easy"},
    {"id": "valid-palindrome",   "title": "Valid Palindrome",   "difficulty": "easy"},
    {"id": "reverse-linked-list","title": "Reverse Linked List","difficulty": "medium"},
    {"id": "fizz-buzz",          "title": "Fizz Buzz",          "difficulty": "easy"},
]

DPP = {
    "mode": "coding_challenge",
    "session": {"elapsed_minutes": 6, "total_problems": 4, "hints_given": 0},
    "live_code": {"language": "python"},
    "candidate": {"first_name": "Zohar", "full_name": "Zohar Babin"},
    "all_problems_in_session": PROBLEMS,
}

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 2
RETRY_BACKOFF_S = 3

def api_call(url: str, payload: dict) -> dict:
    """POST JSON to URL with retry logic matching the client (2 retries, 3s backoff, retry on 503/429)."""
    data = json.dumps(payload).encode()
    t0 = time.perf_counter()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            time.sleep(RETRY_BACKOFF_S)

        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=25) as resp:
                body = json.loads(resp.read())
                elapsed = time.perf_counter() - t0
                return {"ok": True, "status": resp.status, "elapsed": elapsed, "body": body,
                        "attempts": attempt}
        except HTTPError as e:
            last_error = str(e)
            if e.code in (503, 429) and attempt < MAX_RETRIES:
                continue
            elapsed = time.perf_counter() - t0
            return {"ok": False, "status": e.code, "elapsed": elapsed, "body": None,
                    "error": last_error, "attempts": attempt}
        except (URLError, TimeoutError, Exception) as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                continue
            elapsed = time.perf_counter() - t0
            return {"ok": False, "status": 0, "elapsed": elapsed, "body": None,
                    "error": last_error, "attempts": attempt}

    elapsed = time.perf_counter() - t0
    return {"ok": False, "status": 0, "elapsed": elapsed, "body": None,
            "error": last_error or "Max retries exceeded", "attempts": MAX_RETRIES}

# ─────────────────────────────────────────────────────────────────────────────
# Single run: 4 parallel per-problem + 1 synthesis
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(url: str) -> dict:
    """Execute one full iterative pipeline. Returns timing breakdown."""
    run_start = time.perf_counter()

    # ── Phase 1: per-problem (parallel) ──────────────────────────────────
    phase1_start = time.perf_counter()
    per_problem_results = []

    def call_per_problem(problem):
        payload = {
            "analysis_mode": "per_problem",
            "transcript": TRANSCRIPT,
            "problem_focus": problem,
            "dpp": DPP,
        }
        return problem["id"], api_call(url, payload)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(call_per_problem, p): p for p in PROBLEMS}
        for future in as_completed(futures):
            pid, result = future.result()
            per_problem_results.append({"problem_id": pid, **result})

    phase1_elapsed = time.perf_counter() - phase1_start

    # Collect per-problem analyses for synthesis input
    problem_analyses = []
    per_problem_details = []
    for r in per_problem_results:
        pid = r["problem_id"]
        detail = {
            "problem_id": pid,
            "ok": r["ok"],
            "status": r["status"],
            "elapsed_s": round(r["elapsed"], 3),
            "attempts": r.get("attempts", 1),
            "tokens_in": None,
            "tokens_out": None,
        }
        if r["ok"] and r["body"] and r["body"].get("success"):
            problem_analyses.append(r["body"]["summary"])
            usage = r["body"].get("usage", {})
            detail["tokens_in"] = usage.get("input_tokens")
            detail["tokens_out"] = usage.get("output_tokens")
        else:
            detail["error"] = r.get("error", "unknown")
        per_problem_details.append(detail)

    # ── Phase 2: synthesis ───────────────────────────────────────────────
    phase2_start = time.perf_counter()
    synth_payload = {
        "analysis_mode": "synthesis",
        "problem_results": problem_analyses,
        "dpp": DPP,
    }
    synth_result = api_call(url, synth_payload)
    phase2_elapsed = time.perf_counter() - phase2_start

    synth_detail = {
        "ok": synth_result["ok"],
        "status": synth_result["status"],
        "elapsed_s": round(synth_result["elapsed"], 3),
        "attempts": synth_result.get("attempts", 1),
        "tokens_in": None,
        "tokens_out": None,
    }
    if synth_result["ok"] and synth_result["body"] and synth_result["body"].get("success"):
        usage = synth_result["body"].get("usage", {})
        synth_detail["tokens_in"] = usage.get("input_tokens")
        synth_detail["tokens_out"] = usage.get("output_tokens")
    else:
        synth_detail["error"] = synth_result.get("error", "unknown")

    total_elapsed = time.perf_counter() - run_start

    return {
        "total_s": round(total_elapsed, 3),
        "phase1_s": round(phase1_elapsed, 3),
        "phase2_s": round(phase2_elapsed, 3),
        "per_problem": per_problem_details,
        "synthesis": synth_detail,
        "all_ok": all(d["ok"] for d in per_problem_details) and synth_detail["ok"],
    }

# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, unit="s", width=8):
    if val is None:
        return "   N/A".ljust(width)
    return f"{val:>{width}.3f}{unit}"

def print_report(runs: list, threshold: float):
    W = 78
    print("\n" + "=" * W)
    print("  BENCHMARK REPORT — Iterative Analysis Pipeline (4 problems + synthesis)")
    print("=" * W)

    totals      = [r["total_s"]  for r in runs]
    phase1s     = [r["phase1_s"] for r in runs]
    phase2s     = [r["phase2_s"] for r in runs]
    all_ok      = [r["all_ok"]   for r in runs]
    successes   = sum(all_ok)
    failures    = len(runs) - successes
    pass_count  = sum(1 for t in totals if t < threshold)

    # ── Per-run table ────────────────────────────────────────────────────
    print(f"\n{'Run':>4}  {'Total':>8}  {'Phase1':>8}  {'Phase2':>8}  {'Status':>8}  {'Pass':>6}  Per-problem breakdown")
    print("-" * W)
    for i, r in enumerate(runs, 1):
        pp_times = "  ".join(
            f"{d['problem_id'].split('-')[0][:6]}={d['elapsed_s']:.1f}s"
            for d in sorted(r["per_problem"], key=lambda x: x["problem_id"])
        )
        status = "OK" if r["all_ok"] else "FAIL"
        passed = "PASS" if r["total_s"] < threshold else "FAIL"
        print(f"  {i:>2}  {r['total_s']:>7.2f}s  {r['phase1_s']:>7.2f}s  {r['phase2_s']:>7.2f}s  {status:>8}  {passed:>6}  {pp_times}")

    # ── Aggregate stats ──────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  AGGREGATE STATS ({len(runs)} runs)")
    print(f"{'─' * W}")

    def stat_line(label, values):
        mn = min(values)
        mx = max(values)
        avg = statistics.mean(values)
        med = statistics.median(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0
        p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 5 else mx
        print(f"  {label:<18}  min={mn:.2f}s  avg={avg:.2f}s  med={med:.2f}s  p95={p95:.2f}s  max={mx:.2f}s  sd={sd:.2f}s")

    stat_line("End-to-end", totals)
    stat_line("Phase 1 (parallel)", phase1s)
    stat_line("Phase 2 (synthesis)", phase2s)

    # Per-problem aggregate
    print(f"\n  Per-problem call times:")
    for pid in [p["id"] for p in PROBLEMS]:
        times = []
        for r in runs:
            for d in r["per_problem"]:
                if d["problem_id"] == pid:
                    times.append(d["elapsed_s"])
        if times:
            stat_line(f"    {pid}", times)

    # Token stats
    all_pp_in  = [d["tokens_in"]  for r in runs for d in r["per_problem"] if d["tokens_in"]]
    all_pp_out = [d["tokens_out"] for r in runs for d in r["per_problem"] if d["tokens_out"]]
    all_sy_in  = [r["synthesis"]["tokens_in"]  for r in runs if r["synthesis"]["tokens_in"]]
    all_sy_out = [r["synthesis"]["tokens_out"] for r in runs if r["synthesis"]["tokens_out"]]

    print(f"\n  Token usage (avg):")
    if all_pp_in:
        print(f"    Per-problem  in={statistics.mean(all_pp_in):.0f}  out={statistics.mean(all_pp_out):.0f}")
    if all_sy_in:
        print(f"    Synthesis    in={statistics.mean(all_sy_in):.0f}  out={statistics.mean(all_sy_out):.0f}")

    # ── Slowest call analysis ────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  BOTTLENECK ANALYSIS")
    print(f"{'─' * W}")

    slowest_total = max(runs, key=lambda r: r["total_s"])
    si = runs.index(slowest_total) + 1
    print(f"  Slowest run: #{si} at {slowest_total['total_s']:.2f}s")
    print(f"    Phase 1: {slowest_total['phase1_s']:.2f}s  Phase 2: {slowest_total['phase2_s']:.2f}s")
    slowest_pp = max(slowest_total["per_problem"], key=lambda d: d["elapsed_s"])
    print(f"    Slowest per-problem call: {slowest_pp['problem_id']} at {slowest_pp['elapsed_s']:.2f}s")

    avg_p1 = statistics.mean(phase1s)
    avg_p2 = statistics.mean(phase2s)
    pct_p1 = avg_p1 / statistics.mean(totals) * 100
    pct_p2 = avg_p2 / statistics.mean(totals) * 100
    print(f"\n  Avg time split: Phase 1 = {avg_p1:.2f}s ({pct_p1:.0f}%)  Phase 2 = {avg_p2:.2f}s ({pct_p2:.0f}%)")
    bottleneck = "Phase 2 (synthesis)" if avg_p2 > avg_p1 else "Phase 1 (per-problem)"
    print(f"  Bottleneck: {bottleneck}")

    # ── Final verdict ────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    verdict = "PASS" if pass_count == len(runs) else "FAIL"
    color = "\033[92m" if verdict == "PASS" else "\033[91m"
    reset = "\033[0m"
    print(f"  RESULT: {color}{verdict}{reset}  —  {pass_count}/{len(runs)} runs under {threshold}s threshold")
    print(f"  API errors: {failures}/{len(runs)} runs had at least one failed call")
    total_retries = sum(
        (d["attempts"] - 1) for r in runs for d in r["per_problem"]
    ) + sum((r["synthesis"]["attempts"] - 1) for r in runs)
    total_calls = len(runs) * (len(PROBLEMS) + 1)
    if total_retries > 0:
        print(f"  Retries: {total_retries}/{total_calls} calls needed retries")
    avg_total = statistics.mean(totals)
    print(f"  Average end-to-end: {avg_total:.2f}s  (headroom: {threshold - avg_total:.2f}s)")
    print("=" * W + "\n")

    return verdict == "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark iterative analysis pipeline")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API Gateway URL")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of iterations")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_S, help="Pass/fail threshold in seconds")
    parser.add_argument("--json", action="store_true", help="Also write raw results to benchmark_results.json")
    args = parser.parse_args()

    print(f"\nBenchmark config:")
    print(f"  API URL:    {args.url}")
    print(f"  Runs:       {args.runs}")
    print(f"  Threshold:  {args.threshold}s")
    print(f"  Transcript: {len(TRANSCRIPT)} messages, {len(PROBLEMS)} problems")
    print(f"  Date:       {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Warmup call
    print(f"\nWarmup call...", end=" ", flush=True)
    warmup = api_call(args.url, {
        "analysis_mode": "per_problem",
        "transcript": TRANSCRIPT[:4],
        "problem_focus": PROBLEMS[0],
        "dpp": DPP,
    })
    print(f"{'OK' if warmup['ok'] else 'FAIL'} ({warmup['elapsed']:.2f}s)")

    # Run benchmark
    runs = []
    for i in range(1, args.runs + 1):
        print(f"\rRun {i}/{args.runs}...", end="", flush=True)
        result = run_pipeline(args.url)
        runs.append(result)
        status = "ok" if result["all_ok"] else "FAIL"
        print(f"\rRun {i}/{args.runs}: {result['total_s']:.2f}s [{status}]  (p1={result['phase1_s']:.1f}s  p2={result['phase2_s']:.1f}s)")

    # Report
    passed = print_report(runs, args.threshold)

    # Optional JSON dump
    if args.json:
        out_path = "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "config": {"url": args.url, "runs": args.runs, "threshold": args.threshold},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "runs": runs,
            }, f, indent=2)
        print(f"  Raw results written to {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
