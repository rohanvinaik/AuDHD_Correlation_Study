"""PDF report generation

Converts HTML reports to publication-quality PDFs with embedded visualizations.
"""
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import base64
import io

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    warnings.warn(
        "WeasyPrint not available. PDF generation will be limited. "
        "Install with: pip install weasyprint"
    )


@dataclass
class PDFConfig:
    """Configuration for PDF generation"""
    page_size: str = 'A4'  # 'A4', 'Letter', etc.
    orientation: str = 'portrait'  # 'portrait' or 'landscape'
    margin_top: str = '2cm'
    margin_bottom: str = '2cm'
    margin_left: str = '2cm'
    margin_right: str = '2cm'
    embed_images: bool = True
    optimize_size: bool = True
    include_toc: bool = True
    include_page_numbers: bool = True
    font_family: str = 'Arial, sans-serif'
    font_size: str = '11pt'
    line_height: str = '1.6'


def generate_pdf_report(
    html_content: str,
    output_path: Path,
    config: Optional[PDFConfig] = None,
    figures: Optional[Dict[str, Path]] = None,
) -> Path:
    """
    Generate PDF report from HTML content

    Args:
        html_content: HTML string to convert
        output_path: Output PDF path
        config: PDF configuration
        figures: Dictionary of figure paths to embed

    Returns:
        Path to generated PDF
    """
    if not WEASYPRINT_AVAILABLE:
        raise RuntimeError(
            "WeasyPrint is required for PDF generation. "
            "Install with: pip install weasyprint"
        )

    config = config or PDFConfig()

    # Embed figures if provided
    if figures and config.embed_images:
        html_content = _embed_figures(html_content, figures)

    # Add PDF-specific CSS
    pdf_css = _generate_pdf_css(config)

    # Generate PDF
    html = HTML(string=html_content)
    css = CSS(string=pdf_css)

    html.write_pdf(
        str(output_path),
        stylesheets=[css],
    )

    return output_path


def _embed_figures(html_content: str, figures: Dict[str, Path]) -> str:
    """
    Embed figures as base64 data URIs

    Args:
        html_content: HTML content
        figures: Dictionary mapping figure IDs to paths

    Returns:
        HTML with embedded figures
    """
    for fig_id, fig_path in figures.items():
        if not fig_path.exists():
            warnings.warn(f"Figure not found: {fig_path}")
            continue

        # Read image and encode to base64
        with open(fig_path, 'rb') as f:
            img_data = f.read()

        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Determine MIME type
        suffix = fig_path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
        }
        mime_type = mime_types.get(suffix, 'image/png')

        # Create data URI
        data_uri = f"data:{mime_type};base64,{img_base64}"

        # Replace placeholder or img src
        html_content = html_content.replace(
            f'src="{fig_id}"',
            f'src="{data_uri}"'
        )
        html_content = html_content.replace(
            f"src='{fig_id}'",
            f'src="{data_uri}"'
        )
        html_content = html_content.replace(
            str(fig_path),
            data_uri
        )

    return html_content


def _generate_pdf_css(config: PDFConfig) -> str:
    """
    Generate PDF-specific CSS

    Args:
        config: PDF configuration

    Returns:
        CSS string
    """
    css = f"""
    @page {{
        size: {config.page_size} {config.orientation};
        margin-top: {config.margin_top};
        margin-bottom: {config.margin_bottom};
        margin-left: {config.margin_left};
        margin-right: {config.margin_right};

        @bottom-center {{
            content: counter(page) " / " counter(pages);
            font-size: 9pt;
            color: #666;
        }}
    }}

    body {{
        font-family: {config.font_family};
        font-size: {config.font_size};
        line-height: {config.line_height};
        color: #333;
    }}

    h1 {{
        font-size: 24pt;
        margin-top: 0;
        margin-bottom: 12pt;
        page-break-after: avoid;
    }}

    h2 {{
        font-size: 18pt;
        margin-top: 18pt;
        margin-bottom: 9pt;
        page-break-after: avoid;
    }}

    h3 {{
        font-size: 14pt;
        margin-top: 12pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
    }}

    p {{
        margin-top: 0;
        margin-bottom: 9pt;
        text-align: justify;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 12pt 0;
        page-break-inside: avoid;
    }}

    th, td {{
        padding: 6pt 8pt;
        border: 1pt solid #ddd;
        text-align: left;
    }}

    th {{
        background-color: #f5f5f5;
        font-weight: bold;
    }}

    img {{
        max-width: 100%;
        height: auto;
        page-break-inside: avoid;
    }}

    .page-break {{
        page-break-before: always;
    }}

    .no-break {{
        page-break-inside: avoid;
    }}

    figure {{
        margin: 12pt 0;
        page-break-inside: avoid;
    }}

    figcaption {{
        font-size: 9pt;
        font-style: italic;
        color: #666;
        margin-top: 6pt;
    }}

    .toc {{
        page-break-after: always;
    }}

    .toc a {{
        text-decoration: none;
        color: #333;
    }}

    .toc li {{
        margin-bottom: 6pt;
    }}
    """

    if not config.include_page_numbers:
        css += """
        @page {
            @bottom-center {
                content: none;
            }
        }
        """

    return css


def generate_pdf_with_toc(
    html_content: str,
    output_path: Path,
    config: Optional[PDFConfig] = None,
    title: str = "Report",
) -> Path:
    """
    Generate PDF with table of contents

    Args:
        html_content: HTML content
        output_path: Output PDF path
        config: PDF configuration
        title: Report title for TOC

    Returns:
        Path to generated PDF
    """
    config = config or PDFConfig()

    if config.include_toc:
        # Parse headers and generate TOC
        toc_html = _generate_toc(html_content, title)
        html_content = toc_html + html_content

    return generate_pdf_report(html_content, output_path, config)


def _generate_toc(html_content: str, title: str) -> str:
    """
    Generate table of contents from HTML headers

    Args:
        html_content: HTML content
        title: Report title

    Returns:
        TOC HTML
    """
    import re

    # Find all headers
    headers = re.findall(
        r'<h([1-3])[^>]*id=["\']([^"\']+)["\'][^>]*>([^<]+)</h\1>',
        html_content
    )

    if not headers:
        return ""

    toc = f"""
    <div class="toc page-break">
        <h1>Table of Contents</h1>
        <ul>
    """

    for level, header_id, header_text in headers:
        indent = int(level) - 1
        toc += f"""
            <li style="margin-left: {indent * 20}px;">
                <a href="#{header_id}">{header_text}</a>
            </li>
        """

    toc += """
        </ul>
    </div>
    """

    return toc


def split_into_chapters(
    html_content: str,
    output_dir: Path,
    config: Optional[PDFConfig] = None,
) -> List[Path]:
    """
    Split report into chapter PDFs

    Args:
        html_content: HTML content
        output_dir: Output directory
        config: PDF configuration

    Returns:
        List of generated PDF paths
    """
    import re

    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or PDFConfig()

    # Split by h1 headers
    chapters = re.split(r'<h1[^>]*>([^<]+)</h1>', html_content)

    pdf_paths = []

    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i].strip()
        chapter_content = chapters[i + 1]

        # Create full chapter HTML
        chapter_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>{chapter_title}</title>
        </head>
        <body>
            <h1>{chapter_title}</h1>
            {chapter_content}
        </body>
        </html>
        """

        # Generate PDF for chapter
        safe_title = re.sub(r'[^\w\s-]', '', chapter_title).strip().replace(' ', '_')
        chapter_path = output_dir / f"chapter_{i // 2 + 1}_{safe_title}.pdf"

        generate_pdf_report(chapter_html, chapter_path, config)
        pdf_paths.append(chapter_path)

    return pdf_paths


def create_combined_pdf(
    pdf_paths: List[Path],
    output_path: Path,
) -> Path:
    """
    Combine multiple PDFs into one

    Args:
        pdf_paths: List of PDF paths to combine
        output_path: Output PDF path

    Returns:
        Path to combined PDF
    """
    try:
        from PyPDF2 import PdfMerger
    except ImportError:
        raise RuntimeError(
            "PyPDF2 is required for combining PDFs. "
            "Install with: pip install PyPDF2"
        )

    merger = PdfMerger()

    for pdf_path in pdf_paths:
        if pdf_path.exists():
            merger.append(str(pdf_path))
        else:
            warnings.warn(f"PDF not found: {pdf_path}")

    merger.write(str(output_path))
    merger.close()

    return output_path


def optimize_pdf_size(
    input_path: Path,
    output_path: Optional[Path] = None,
    quality: str = 'medium',
) -> Path:
    """
    Optimize PDF file size

    Args:
        input_path: Input PDF path
        output_path: Output PDF path (overwrites input if None)
        quality: Quality level ('low', 'medium', 'high')

    Returns:
        Path to optimized PDF
    """
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        raise RuntimeError(
            "PyPDF2 is required for PDF optimization. "
            "Install with: pip install PyPDF2"
        )

    output_path = output_path or input_path

    reader = PdfReader(str(input_path))
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    # Compress
    quality_settings = {
        'low': 0,
        'medium': 1,
        'high': 2,
    }

    for page in writer.pages:
        page.compress_content_streams()

    with open(output_path, 'wb') as f:
        writer.write(f)

    return output_path