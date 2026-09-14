"""Run setup, retrieval, both agents and a repeatable live evaluation."""

from __future__ import annotations

import argparse
import json
import sys

from goodmem.errors import GoodMemError
from langchain_core.messages import HumanMessage

from .agents import answer_text, build_agentic_graph, build_react_agent, tool_calls
from .config import Settings, chat_model
from .ingestion import setup
from .retrieval import make_tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    provision = commands.add_parser("setup", help="Provision spaces and ingest the six official pages")
    provision.add_argument("--init", action="store_true", help="Initialize a new local server")
    provision.add_argument("--with-reranker", action="store_true", help="Register a Cohere reranker")
    retrieve = commands.add_parser("search", help="Retrieve cited chunks directly from GoodMem")
    retrieve.add_argument("collection", choices=["langgraph", "langchain"])
    retrieve.add_argument("question")
    retrieve.add_argument("--no-rerank", action="store_true")
    ask = commands.add_parser("ask", help="Ask a documentation question")
    ask.add_argument("question")
    ask.add_argument("--agent", choices=["graph", "react"], default="react")
    evaluate = commands.add_parser("evaluate", help="Run live retrieval and agent acceptance cases")
    evaluate.add_argument("--output", default=".runtime/evaluation.json")
    evaluate.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.command == "setup":
            state = setup(settings, initialize=args.init, with_reranker=args.with_reranker)
            print(f"Ready: {len(state['documents'])} documents in {len(state['spaces'])} spaces.")
            return
        state = settings.state()
        with settings.client() as client:
            if args.command == "search":
                tools = make_tools(client, state, rerank=not args.no_rerank)
                selected = next(t for t in tools if t.name == f"{args.collection}_docs_tool")
                print(selected.invoke({"query": args.question}) or "No relevant documents found.")
            elif args.command == "ask":
                tools = make_tools(client, state)
                builder = build_agentic_graph if args.agent == "graph" else build_react_agent
                result = builder(chat_model(), tools).invoke(
                    {"messages": [HumanMessage(content=args.question)]}, {"recursion_limit": 60}
                )
                print(answer_text(result))
                print("\nTool calls:", json.dumps(tool_calls(result)), file=sys.stderr)
            else:
                from .evaluation import evaluate

                passed = evaluate(client, settings, state, args.output, args.retrieval_only)
                if not passed:
                    sys.exit(1)
    except (GoodMemError, ValueError, RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
