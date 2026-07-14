#!/usr/bin/env python3
"""
HTTP server with WASM support + Gemini-powered retrieval API.
Run from the obj_viewer directory: python3 serve.py
"""

import http.server
import socketserver
import os
import json
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8080
DIRECTORY = "web"

# ---------------------------------------------------------------------------
# Retrieval backend
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
VLM_MODEL = "gemini-3.1-flash-lite-preview"
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 768

# Paths to pre-computed embeddings and descriptions
ROOT = Path(__file__).resolve().parent.parent
EMBED_DIR = ROOT / "processed_data" / "gemini_embeddings_v2_v7_ifc_aware" / "gemini_embed_single"
DESC_DIR = ROOT / "processed_data" / "descriptions" / "gemini_v2_v7_ifc_aware"
METADATA_PATH = ROOT / "data" / "metadata.json"

# System prompt for generating search descriptions (matches v7 format)
SEARCH_PROMPT = """\
You are a 3D BIM object search assistant. Given a user's search query, generate a \
description that matches how objects are described in our database.

Our database describes objects in this format:
[Thinness], [Shape Class], [Aspect Ratio], [Profile], [Visual Cue]. [keywords]

Where:
- Thinness: Wire-Thin, Very-Thin, Thin, Medium, Thick
- Shape Class: Horizontal-Rod, Vertical-Rod, Diagonal-Rod, Horizontal-Sheet, \
Vertical-Sheet, Flat-Sheet, Block, Complex
- Aspect Ratio: 1:1:1, 1:1:3, 1:1:10, 1:1:30, 1:3:3, 1:3:10, 1:3:30, 1:10:10, 1:10:30, 1:10:100
- Profile: I-Section, T-Section, L-Section, C-Channel, Rectangular, Circular, \
Elliptical, Annular, Triangular, Irregular, Composite
- Visual Cue: free-form 1-3 word visual feature (e.g. "stepped-treads", "glazed-grid", \
"open-balustrade", "globe-on-arm", "scattered-cutouts", "uniform-extrusion")
- Keywords: 1-5 comma-separated functional/geometric keywords (e.g. chair, toilet, \
railing, balustrade, staircase, sink, lamp, pipe, bracket)

BIM object types in our database include:
- Structural: beams (horizontal rods), columns (vertical rods), footings (blocks), \
members (wire-thin bracing/mullions)
- Enclosure: walls (panels with cutouts), slabs (horizontal sheets), roofs (angled/sloped), \
plates (thin panels), curtain walls (glazed grid facades)
- Openings: doors (frame+panel), windows (glazed frame)
- Circulation: stairs (stepped treads), railings (open balustrade frameworks)
- MEP: sinks, toilets, pipes, ducts, fixtures
- Furnishing: chairs, tables, plants, shelving
- Lighting: lamps, ceiling panels, mounted fixtures

Generate a description in the SAME format that best matches what the user is looking for. \
Output exactly one line: 5 comma-separated tags, period, then 1-5 comma-separated keywords.

User query: {query}"""


class RetrievalIndex:
    """Pre-loaded embedding index for nearest neighbor search."""

    def __init__(self):
        self.gids = []
        self.embeddings = None
        self.descriptions = {}
        self.metadata = {}
        self._client = None

    def load(self):
        print("Loading retrieval index...")

        # Load embeddings
        gids, embs = [], []
        if EMBED_DIR.exists():
            for f in sorted(EMBED_DIR.glob("*.npy")):
                gids.append(f.stem)
                embs.append(np.load(f))
            if embs:
                self.gids = gids
                self.embeddings = np.stack(embs)
                # Normalize for cosine similarity
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self.embeddings = self.embeddings / norms
                print(f"  Loaded {len(gids)} embeddings ({self.embeddings.shape})")
            else:
                print(f"  WARNING: No embeddings found in {EMBED_DIR}")
        else:
            print(f"  WARNING: Embedding dir not found: {EMBED_DIR}")

        # Load descriptions
        if DESC_DIR.exists():
            for f in DESC_DIR.glob("*.txt"):
                self.descriptions[f.stem] = f.read_text().strip()
            print(f"  Loaded {len(self.descriptions)} descriptions")

        # Load metadata
        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                meta_list = json.load(f)
            for m in meta_list:
                self.metadata[m.get("GlobalId", "")] = m
            print(f"  Loaded {len(self.metadata)} metadata entries")

    def get_client(self):
        if self._client is None:
            if not API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Export it before starting the server:\n"
                    "    export GEMINI_API_KEY=your-key-here"
                )
            from google import genai
            self._client = genai.Client(api_key=API_KEY)
        return self._client

    def generate_description(self, query: str) -> str:
        """Use Gemini to convert user query into a database-matching description."""
        from google.genai import types
        client = self.get_client()
        prompt = SEARCH_PROMPT.format(query=query)
        response = client.models.generate_content(
            model=VLM_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            ),
        )
        return response.text.strip()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using Gemini Embedding."""
        from google.genai import types
        client = self.get_client()
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBED_DIM,
            ),
        )
        emb = np.array(result.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def search(self, query: str, top_k: int = 20, mode: str = "direct") -> dict:
        """Search pipeline. Modes: 'direct' = embed query as-is, 'generate' = LLM rewrites first."""
        if self.embeddings is None:
            return {"error": "No embeddings loaded"}

        generated_desc = None
        if mode == "generate":
            # LLM rewrites query to match database style, then embed that
            generated_desc = self.generate_description(query)
            query_emb = self.embed_text(generated_desc)
        else:
            # Embed raw query directly — let the embedding model handle similarity
            query_emb = self.embed_text(query)

        # Step 3: Cosine similarity (embeddings already normalized)
        similarities = self.embeddings @ query_emb

        # Step 4: Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            gid = self.gids[idx]
            meta = self.metadata.get(gid, {})
            results.append({
                "global_id": gid,
                "similarity": float(similarities[idx]),
                "description": self.descriptions.get(gid, ""),
                "ifc_type": meta.get("IfcType", ""),
                "mesh_filename": meta.get("mesh_filename", f"{gid}.obj"),
            })

        return {
            "query": query,
            "generated_description": generated_desc,
            "results": results,
        }


# Global index
retrieval_index = RetrievalIndex()


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def guess_type(self, path):
        if path.endswith('.wasm'):
            return 'application/wasm'
        if path.endswith('.js'):
            return 'application/javascript'
        return super().guess_type(path)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/status':
            self._json_response({
                "embeddings_loaded": retrieval_index.embeddings is not None,
                "num_embeddings": len(retrieval_index.gids),
                "num_descriptions": len(retrieval_index.descriptions),
            })
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/search':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                query = data.get("query", "")
                top_k = data.get("top_k", 20)
                mode = data.get("mode", "direct")
                result = retrieval_index.search(query, top_k, mode)
                self._json_response(result)
            except Exception as e:
                self._json_response({"error": str(e)}, status=500)
            return

        self.send_error(404)

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress static file logs, show API logs
        if '/api/' in str(args[0]) if args else False:
            super().log_message(format, *args)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    retrieval_index.load()

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\nServing OBJ Viewer at http://localhost:{PORT}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
