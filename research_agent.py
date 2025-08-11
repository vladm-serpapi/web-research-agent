#!/usr/bin/env python3

import typing as t
import argparse, json, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from serpapi import GoogleSearch


class ResearchAgent:
    """LLM‑powered researcher that combines OpenAI o‑series model with SerpAPI."""

    def __init__(
        self,
        model: str = "o3",
        topn: int = 10,
        debug: bool = False,
        openai_key: t.Optional[str] = None,
        serpapi_key: t.Optional[str] = None,
    ) -> None:
        self.model = model
        self.topn = topn
        self.debug = debug
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.serp_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        if not self.openai_key or not self.serp_key:
            raise RuntimeError("OPENAI_API_KEY and SERPAPI_API_KEY must be set.")

        self.client = OpenAI(api_key=self.openai_key)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search Google and return the top result snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Google search string",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        self.sys_prompt = (
            "You are a meticulous research assistant.\n"
            "When outside knowledge is needed, you must emit ALL `search_web` tool calls "
            "in a SINGLE assistant message before reading any results.\n\n"
            "You must return them in the exact JSON structure the API expects for `tool_calls`,\n"
            "with each having its own `id`, `type`, and `function` fields.\n"
            "Do not write explanations, just the tool calls.\n\n"
            "For example:\n"
            "{\n"
            "  \"tool_calls\": [\n"
            "    {\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"search_web\", \"arguments\": \"{\\\"query\\\": \\\"first topic\\\"}\"}},\n"
            "    {\"id\": \"call_2\", \"type\": \"function\", \"function\": {\"name\": \"search_web\", \"arguments\": \"{\\\"query\\\": \\\"second topic\\\"}\"}}\n"
            "  ]\n"
            "}\n\n"
            "Always batch between 2 and 50 calls in a single turn if you need external data.\n"
            "Only after all tool outputs are returned should you write your final, well-cited answer."
        )

    def _search_web(self, query: str) -> str:
        if self.debug:
            print(f"[DEBUG] → SerpAPI query: '{query}'")
        search = GoogleSearch({"q": query, "api_key": self.serp_key, "num": self.topn})
        org = search.get_dict().get("organic_results", [])[: self.topn]
        return "".join(
            f"- {r.get('title','(untitled)')}: {r.get('snippet','(no snippet)')}" for r in org
        ) or "No results found."

    def run(self, question: str) -> dict[str, t.Any]:
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": question},
        ]
        steps: list[dict[str, t.Any]] = []

        while True:
            if self.debug:
                print("[DEBUG] → OpenAI chat.completions.create request …")

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                # append assistant message FIRST (per API contract)
                messages.append(msg)

                # fetch all tool results concurrently
                def fetch(call):
                    args = json.loads(call.function.arguments)
                    q = args["query"]
                    steps.append({"type": "tool_call", "query": q})
                    return call.id, q, self._search_web(q)

                with ThreadPoolExecutor() as pool:
                    results = list(pool.map(fetch, msg.tool_calls))

                # append each tool result in the SAME ORDER as tool_calls
                for call_id, q, result in results:
                    steps.append({"type": "tool_result", "content": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result,
                        }
                    )
                continue

            # no tool calls → final answer
            answer = msg.content.strip()
            steps.append({"type": "assistant_answer", "content": answer})
            result: dict[str, t.Any] = {"question": question, "answer": answer, "steps": steps}
            return result


# -------------------------------------------------------------------------
# CLI wrapper
# -------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="ResearchAgent CLI")
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-m", "--model", default="gpt-4o", choices=["o3", "o4-mini", "gpt-4o"])
    p.add_argument("-n", "--topn", type=int, default=10)
    p.add_argument("-o", "--outfile", type=Path)
    p.add_argument("-d", "--debug", action="store_true")
    cfg = p.parse_args()

    agent = ResearchAgent(model=cfg.model, topn=cfg.topn, debug=cfg.debug)
    result = agent.run(cfg.query)

    print("" + "=" * 80)
    print(result["answer"])
    print("=" * 80)

    if cfg.outfile:
        cfg.outfile.write_text(json.dumps(result, indent=2))
        print(f"Saved full trace → {cfg.outfile}")


if __name__ == "__main__":
    _cli()