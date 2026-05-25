#!/usr/bin/env python3
"""
dots.mocr Parser for PDF Document Processing

Uses dots.mocr (Vision-Language Model) for advanced document layout parsing
with multilingual OCR support.

GitHub: https://github.com/rednote-hilab/dots.ocr

Features:
- Multilingual text recognition
- Document layout analysis (headers, tables, figures)
- SVG conversion for graphics
- Clean markdown output

Usage:
    python dots_ocr_parser.py --input-dir ../market_data/Investing_Books/ --output-dir ./data/extracted_text/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    os.system("pip install pillow")
    from PIL import Image

try:
    import pymupdf
except ImportError:
    os.system("pip install pymupdf")
    import pymupdf


class DotsMocrParser:
    """
    Document parser using dots.mocr Vision-Language Model.

    Supports two modes:
    1. Local vLLM server (recommended for performance)
    2. Hugging Face Transformers (slower, no GPU required)
    """

    def __init__(
        self,
        mode: str = "vllm",
        vllm_base_url: str = "http://localhost:8000/v1",
        model_name: str = "rednote-hilab/dots.mocr",
        api_key: Optional[str] = None,
    ):
        """
        Initialize dots.mocr parser.

        Args:
            mode: "vllm" or "huggingface"
            vllm_base_url: vLLM server URL (OpenAI-compatible API)
            model_name: Model name for HuggingFace mode
            api_key: API key for vLLM server (if required)
        """
        self.mode = mode
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or "not-needed"

        if mode == "huggingface":
            self._init_huggingface()

    def _init_huggingface(self):
        """Initialize HuggingFace Transformers backend."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch
        except ImportError:
            print("Installing required packages...")
            os.system("pip install transformers torch accelerate")
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _pdf_page_to_image(self, pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
        """Convert PDF page to PIL Image."""
        doc = pymupdf.open(pdf_path)
        page = doc[page_num]

        # Render page to image
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    def parse_image(self, image: Image.Image, prompt: str = "Convert this document to markdown") -> Dict[str, Any]:
        """
        Parse a single image using dots.mocr.

        Args:
            image: PIL Image of the document
            prompt: Prompt for the model

        Returns:
            Dict with markdown, layout_json, and svg_elements
        """
        if self.mode == "vllm":
            return self._parse_vllm(image, prompt)
        else:
            return self._parse_huggingface(image, prompt)

    def _parse_vllm(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Parse using vLLM server (OpenAI-compatible API)."""
        import requests

        image_base64 = self._image_to_base64(image)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 4096,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        response = requests.post(
            f"{self.vllm_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(f"vLLM API error: {response.status_code} - {response.text}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return {
            "markdown": content,
            "layout_json": None,
            "svg_elements": [],
        }

    def _parse_huggingface(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Parse using HuggingFace Transformers."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
            )

        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return {
            "markdown": generated_text,
            "layout_json": None,
            "svg_elements": [],
        }

    def parse_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        dpi: int = 150,
        prompt: str = "Convert this document to markdown",
    ) -> Dict[str, Any]:
        """
        Parse a PDF file.

        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to parse (None = all pages)
            dpi: Rendering DPI
            prompt: Prompt for the model

        Returns:
            Dict with full document content
        """
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)

        if pages is None:
            pages = list(range(total_pages))
        else:
            pages = [p for p in pages if p < total_pages]

        results = []
        full_markdown = []

        for page_num in pages:
            image = self._pdf_page_to_image(pdf_path, page_num, dpi)
            result = self.parse_image(image, prompt)

            results.append({
                "page": page_num + 1,
                "markdown": result["markdown"],
            })
            full_markdown.append(f"--- Page {page_num + 1} ---\n\n{result['markdown']}")

        doc.close()

        return {
            "file": pdf_path,
            "total_pages": total_pages,
            "parsed_pages": len(pages),
            "pages": results,
            "full_markdown": "\n\n".join(full_markdown),
        }


def process_pdf_with_dots(
    pdf_path: Path,
    output_dir: Path,
    parser: DotsMocrParser,
    logger,
) -> Optional[Dict[str, Any]]:
    """Process a single PDF with dots.mocr."""
    try:
        logger.info(f"Processing {pdf_path.name}...")

        result = parser.parse_pdf(str(pdf_path))

        # Save markdown
        output_file = output_dir / f"{pdf_path.stem}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Source: {pdf_path}\n\n")
            f.write(result["full_markdown"])

        stats = {
            "file": pdf_path.name,
            "total_pages": result["total_pages"],
            "parsed_pages": result["parsed_pages"],
            "output": str(output_file),
            "characters": len(result["full_markdown"]),
        }

        logger.info(f"  Saved: {output_file.name} ({len(result['full_markdown']) / 1024 / 1024:.1f} MB)")
        return stats

    except Exception as e:
        logger.error(f"  ERROR: {type(e).__name__} - {e}")
        return {"file": pdf_path.name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Parse PDFs using dots.mocr OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using vLLM server (recommended)
  python dots_ocr_parser.py --input-dir ../market_data/Investing_Books/ --mode vllm

  # Using HuggingFace (slower, no GPU needed)
  python dots_ocr_parser.py --input-dir ../market_data/Investing_Books/ --mode huggingface
        """
    )

    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with PDFs")
    parser.add_argument("--output-dir", type=str, default="./data/extracted_text_dots", help="Output directory")
    parser.add_argument("--mode", type=str, choices=["vllm", "huggingface"], default="vllm",
                        help="Backend mode")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--model", type=str, default="rednote-hilab/dots.mocr",
                        help="Model name")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    import logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logger = logging.getLogger("dots_ocr_parser")

    # Initialize parser
    logger.info(f"Initializing dots.mocr parser (mode={args.mode})...")
    dots_parser = DotsMocrParser(
        mode=args.mode,
        vllm_base_url=args.vllm_url,
        model_name=args.model,
    )

    # Setup directories
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find PDFs
    pdf_files = sorted(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {args.input_dir}")
        return

    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]
        logger.info(f"Processing first {len(pdf_files)} files (limit={args.limit})")

    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info("")

    # Process PDFs
    stats = {"processed": 0, "errors": 0, "total_pages": 0, "total_chars": 0}

    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] {pdf_file.name}")
        result = process_pdf_with_dots(pdf_file, output_path, dots_parser, logger)

        if result and "error" not in result:
            stats["processed"] += 1
            stats["total_pages"] += result.get("total_pages", 0)
            stats["total_chars"] += result.get("characters", 0)
        else:
            stats["errors"] += 1

        logger.info("")

    logger.info("=" * 50)
    logger.info("dots.mocr Processing Complete")
    logger.info(f"PDFs processed: {stats['processed']}/{len(pdf_files)}")
    logger.info(f"Total pages: {stats['total_pages']}")
    logger.info(f"Total characters: {stats['total_chars']:,}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info("")


if __name__ == "__main__":
    main()
