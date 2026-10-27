#!/usr/bin/env bash
# PostToolUse hook (Edit|Write): rebuild + verify bibliography when a .bib file under literature/ changes.
# Exit 2 sends stderr back to Claude so it fixes the problem.
set -u
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
f=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))')
case "$f" in
  */literature/bib/*.bib) ;;
  */literature/references.bib)
    echo "literature/references.bib is generated; edit literature/bib/*.bib instead. Your edit was overwritten by the rebuild." >&2
    python3 "$ROOT/scripts/merge_bib.py" >/dev/null; exit 2 ;;
  *) exit 0 ;;
esac
out=$(python3 "$ROOT/scripts/merge_bib.py" 2>&1) || { echo "merge_bib.py failed: $out" >&2; exit 2; }
ver=$(python3 "$ROOT/scripts/verify_bib.py" 2>&1) || { echo "Bibliography check failed after editing $f:"$'\n'"$ver"$'\n'"Fix or remove the flagged entries (resolve via arXiv/DOI, never from memory)." >&2; exit 2; }
echo "bib ok: $(echo "$out" | head -1); $(echo "$ver" | tail -1)"
