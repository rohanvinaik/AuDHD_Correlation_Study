"""HTML report generation

Creates standalone HTML reports with embedded assets for offline viewing.
"""
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import base64
import warnings
import re


@dataclass
class HTMLConfig:
    """Configuration for HTML generation"""
    embed_images: bool = True
    embed_css: bool = True
    embed_fonts: bool = False
    minify: bool = False
    include_search: bool = True
    include_print_styles: bool = True
    theme: str = 'light'  # 'light', 'dark', 'auto'
    enable_interactive: bool = True
    add_navigation: bool = True
    responsive_design: bool = True


def generate_html_report(
    html_content: str,
    output_path: Path,
    config: Optional[HTMLConfig] = None,
    figures: Optional[Dict[str, Path]] = None,
    title: str = "Report",
) -> Path:
    """
    Generate standalone HTML report

    Args:
        html_content: HTML content
        output_path: Output HTML path
        config: HTML configuration
        figures: Dictionary of figure paths to embed
        title: Report title

    Returns:
        Path to generated HTML
    """
    config = config or HTMLConfig()

    # Embed figures
    if figures and config.embed_images:
        html_content = _embed_figures_html(html_content, figures)

    # Add enhancements
    if config.add_navigation:
        html_content = _add_navigation(html_content)

    if config.include_search:
        html_content = _add_search_functionality(html_content)

    # Wrap in complete HTML document
    full_html = _create_standalone_html(
        html_content,
        title=title,
        config=config,
    )

    # Minify if requested
    if config.minify:
        full_html = _minify_html(full_html)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_html, encoding='utf-8')

    return output_path


def _embed_figures_html(html_content: str, figures: Dict[str, Path]) -> str:
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

        # Read and encode image
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
            '.webp': 'image/webp',
        }
        mime_type = mime_types.get(suffix, 'image/png')

        data_uri = f"data:{mime_type};base64,{img_base64}"

        # Replace all references
        html_content = html_content.replace(f'src="{fig_id}"', f'src="{data_uri}"')
        html_content = html_content.replace(f"src='{fig_id}'", f'src="{data_uri}"')
        html_content = html_content.replace(str(fig_path), data_uri)

    return html_content


def _create_standalone_html(
    content: str,
    title: str,
    config: HTMLConfig,
) -> str:
    """
    Create complete standalone HTML document

    Args:
        content: Main content HTML
        title: Document title
        config: HTML configuration

    Returns:
        Complete HTML document
    """
    # Base CSS
    base_css = _generate_base_css(config)

    # Print styles
    print_css = _generate_print_css() if config.include_print_styles else ""

    # Theme handling
    theme_css = _generate_theme_css(config.theme)

    # Interactive features
    interactive_js = _generate_interactive_js() if config.enable_interactive else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="AuDHD Correlation Study Report Generator">
    <title>{title}</title>
    <style>
        {base_css}
        {theme_css}
        {print_css}
    </style>
</head>
<body>
    <div id="report-container">
        {content}
    </div>
    <script>
        {interactive_js}
    </script>
</body>
</html>"""

    return html


def _generate_base_css(config: HTMLConfig) -> str:
    """Generate base CSS styles"""
    css = """
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
        background-color: #fff;
        padding: 20px;
    }

    #report-container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        padding: 40px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    h1 {
        font-size: 2.5em;
        margin-bottom: 0.5em;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.3em;
    }

    h2 {
        font-size: 2em;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        color: #34495e;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.3em;
    }

    h3 {
        font-size: 1.5em;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
        color: #34495e;
    }

    p {
        margin-bottom: 1em;
        text-align: justify;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5em 0;
        font-size: 0.95em;
    }

    th, td {
        padding: 12px;
        text-align: left;
        border: 1px solid #ddd;
    }

    th {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.9em;
    }

    tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    tr:hover {
        background-color: #e9ecef;
    }

    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1.5em auto;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    figure {
        margin: 2em 0;
    }

    figcaption {
        font-size: 0.9em;
        font-style: italic;
        color: #666;
        text-align: center;
        margin-top: 0.5em;
    }

    ul, ol {
        margin-left: 2em;
        margin-bottom: 1em;
    }

    li {
        margin-bottom: 0.5em;
    }

    blockquote {
        border-left: 4px solid #3498db;
        padding-left: 1em;
        margin: 1.5em 0;
        font-style: italic;
        color: #555;
    }

    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }

    pre {
        background-color: #f4f4f4;
        padding: 1em;
        border-radius: 4px;
        overflow-x: auto;
        margin: 1em 0;
    }

    pre code {
        background: none;
        padding: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
    }

    .metric-label {
        font-size: 1em;
        opacity: 0.9;
    }

    .alert {
        padding: 15px;
        margin: 1em 0;
        border-radius: 4px;
        border-left: 4px solid;
    }

    .alert-info {
        background-color: #d1ecf1;
        border-color: #0c5460;
        color: #0c5460;
    }

    .alert-warning {
        background-color: #fff3cd;
        border-color: #856404;
        color: #856404;
    }

    .alert-danger {
        background-color: #f8d7da;
        border-color: #721c24;
        color: #721c24;
    }

    .nav-toc {
        position: sticky;
        top: 20px;
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 2em;
    }

    .nav-toc ul {
        list-style: none;
        margin-left: 0;
    }

    .nav-toc a {
        color: #3498db;
        text-decoration: none;
        transition: color 0.2s;
    }

    .nav-toc a:hover {
        color: #2980b9;
        text-decoration: underline;
    }

    .search-box {
        position: sticky;
        top: 0;
        background: white;
        padding: 15px;
        border-bottom: 2px solid #3498db;
        margin-bottom: 2em;
        z-index: 100;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .search-box input {
        width: 100%;
        padding: 10px;
        font-size: 1em;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .search-box input:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 5px rgba(52, 152, 219, 0.3);
    }

    .highlight {
        background-color: yellow;
        font-weight: bold;
    }
    """

    if config.responsive_design:
        css += """
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }

            #report-container {
                padding: 20px;
            }

            h1 {
                font-size: 2em;
            }

            h2 {
                font-size: 1.5em;
            }

            h3 {
                font-size: 1.2em;
            }

            table {
                font-size: 0.85em;
            }

            th, td {
                padding: 8px;
            }
        }
        """

    return css


def _generate_theme_css(theme: str) -> str:
    """Generate theme-specific CSS"""
    if theme == 'dark':
        return """
        body {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }

        #report-container {
            background-color: #2d2d2d;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }

        h1, h2, h3 {
            color: #ffffff;
        }

        h1 {
            border-bottom-color: #4a9eff;
        }

        h2 {
            border-bottom-color: #3a3a3a;
        }

        th {
            background-color: #4a9eff;
        }

        tr:nth-child(even) {
            background-color: #3a3a3a;
        }

        tr:hover {
            background-color: #4a4a4a;
        }

        code, pre {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }

        .nav-toc {
            background: #3a3a3a;
        }

        .search-box {
            background: #2d2d2d;
            border-bottom-color: #4a9eff;
        }

        .search-box input {
            background: #3a3a3a;
            color: #e0e0e0;
            border-color: #4a4a4a;
        }
        """
    elif theme == 'auto':
        return """
        @media (prefers-color-scheme: dark) {
            body {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }

            #report-container {
                background-color: #2d2d2d;
            }

            h1, h2, h3 {
                color: #ffffff;
            }
        }
        """
    else:
        return ""


def _generate_print_css() -> str:
    """Generate print-specific CSS"""
    return """
    @media print {
        body {
            background: white;
            color: black;
        }

        #report-container {
            box-shadow: none;
            padding: 0;
        }

        .search-box {
            display: none;
        }

        .nav-toc {
            page-break-after: always;
        }

        h1, h2, h3 {
            page-break-after: avoid;
        }

        table, figure, img {
            page-break-inside: avoid;
        }

        a {
            color: black;
            text-decoration: none;
        }

        @page {
            margin: 2cm;
        }
    }
    """


def _generate_interactive_js() -> str:
    """Generate JavaScript for interactive features"""
    return """
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Back to top button
    const backToTopBtn = document.createElement('button');
    backToTopBtn.innerHTML = '↑ Top';
    backToTopBtn.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 10px 20px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        display: none;
        z-index: 1000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: opacity 0.3s;
    `;
    document.body.appendChild(backToTopBtn);

    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTopBtn.style.display = 'block';
        } else {
            backToTopBtn.style.display = 'none';
        }
    });

    backToTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Table sorting
    document.querySelectorAll('table').forEach(table => {
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                sortTable(table, index);
            });
        });
    });

    function sortTable(table, column) {
        const tbody = table.querySelector('tbody');
        if (!tbody) return;

        const rows = Array.from(tbody.querySelectorAll('tr'));
        const sortedRows = rows.sort((a, b) => {
            const aVal = a.querySelectorAll('td')[column]?.textContent || '';
            const bVal = b.querySelectorAll('td')[column]?.textContent || '';

            // Try numeric comparison
            const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
            const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

            if (!isNaN(aNum) && !isNaN(bNum)) {
                return aNum - bNum;
            }

            // String comparison
            return aVal.localeCompare(bVal);
        });

        sortedRows.forEach(row => tbody.appendChild(row));
    }

    // Image zoom on click
    document.querySelectorAll('img').forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', () => {
            const overlay = document.createElement('div');
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.9);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
            `;

            const zoomedImg = document.createElement('img');
            zoomedImg.src = img.src;
            zoomedImg.style.cssText = `
                max-width: 90%;
                max-height: 90%;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            `;

            overlay.appendChild(zoomedImg);
            document.body.appendChild(overlay);

            overlay.addEventListener('click', () => {
                document.body.removeChild(overlay);
            });
        });
    });
    """


def _add_navigation(html_content: str) -> str:
    """Add navigation menu based on headers"""
    # Extract headers
    import re
    headers = re.findall(
        r'<h([1-3])[^>]*>([^<]+)</h\1>',
        html_content
    )

    if not headers:
        return html_content

    # Generate IDs for headers if not present
    processed_content = html_content
    nav_items = []

    for i, (level, text) in enumerate(headers):
        header_id = f"section-{i}"
        nav_items.append((int(level), text, header_id))

        # Add ID to header if not present
        old_header = f"<h{level}>{text}</h{level}>"
        new_header = f'<h{level} id="{header_id}">{text}</h{level}>'
        processed_content = processed_content.replace(old_header, new_header, 1)

    # Build navigation
    nav_html = '<nav class="nav-toc"><h3>Contents</h3><ul>'

    for level, text, header_id in nav_items:
        indent = (level - 1) * 20
        nav_html += f'<li style="margin-left: {indent}px;"><a href="#{header_id}">{text}</a></li>'

    nav_html += '</ul></nav>'

    # Insert navigation at the beginning
    return nav_html + processed_content


def _add_search_functionality(html_content: str) -> str:
    """Add search box and functionality"""
    search_html = r"""
    <div class="search-box">
        <input type="text" id="search-input" placeholder="Search report..." />
    </div>
    <script>
    document.getElementById('search-input').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const content = document.getElementById('report-container');

        // Remove previous highlights
        content.innerHTML = content.innerHTML.replace(/<mark class="highlight">([^<]+)<\/mark>/gi, '$1');

        if (searchTerm.length < 3) return;

        // Highlight matches
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        const walker = document.createTreeWalker(
            content,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const nodesToReplace = [];
        while (walker.nextNode()) {
            const node = walker.currentNode;
            if (node.parentElement.tagName !== 'SCRIPT' &&
                node.parentElement.tagName !== 'STYLE' &&
                regex.test(node.textContent)) {
                nodesToReplace.push(node);
            }
        }

        nodesToReplace.forEach(node => {
            const span = document.createElement('span');
            span.innerHTML = node.textContent.replace(regex, '<mark class="highlight">$1</mark>');
            node.parentNode.replaceChild(span, node);
        });
    });
    </script>
    """

    return search_html + html_content


def _minify_html(html: str) -> str:
    """Basic HTML minification"""
    import re

    # Remove comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Remove excess whitespace
    html = re.sub(r'\s+', ' ', html)

    # Remove whitespace between tags
    html = re.sub(r'>\s+<', '><', html)

    return html.strip()


def generate_interactive_html(
    html_content: str,
    output_path: Path,
    figures: Optional[Dict[str, Path]] = None,
    enable_plotly: bool = True,
) -> Path:
    """
    Generate interactive HTML with JavaScript charts

    Args:
        html_content: HTML content
        output_path: Output path
        figures: Figure paths
        enable_plotly: Include Plotly.js for interactive plots

    Returns:
        Path to generated HTML
    """
    config = HTMLConfig(
        enable_interactive=True,
        add_navigation=True,
        include_search=True,
    )

    if enable_plotly:
        # Add Plotly CDN
        plotly_script = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        html_content = plotly_script + html_content

    return generate_html_report(
        html_content,
        output_path,
        config,
        figures,
    )