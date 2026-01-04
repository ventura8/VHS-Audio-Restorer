import xml.etree.ElementTree as ET
import sys
import os
from datetime import datetime

# Import radon for direct complexity calculation
try:
    from radon.complexity import cc_visit
except ImportError:
    print("Error: radon is not installed. Please install it with: pip install radon")
    sys.exit(1)


def generate_badge(line_rate, output_path="assets/coverage.svg"):
    try:
        coverage = float(line_rate) * 100
    except ValueError:
        coverage = 0.0

    color = "#e05d44"  # red
    if coverage >= 95:
        color = "#4c1"  # brightgreen
    elif coverage >= 90:
        color = "#97ca00"  # green
    elif coverage >= 75:
        color = "#dfb317"  # yellow
    elif coverage >= 50:
        color = "#fe7d37"  # orange

    label_text = "Coverage"
    value_text = f"{coverage:.0f}%"

    # Ensure assets dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Estimate widths
    label_width = 70
    value_width = int(len(value_text) * 8.5) + 12

    total_width = label_width + value_width

    # Center positions
    label_x = label_width / 2.0 * 10
    value_x = (label_width + value_width / 2.0) * 10

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" \
height="20" role="img" aria-label="{label_text}: {value_text}">
    <title>{label_text}: {value_text}</title>
    <linearGradient id="s" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="r">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#r)">
        <rect width="{label_width}" height="20" fill="#555"/>
        <rect x="{label_width}" width="{value_width}" height="20" \
fill="{color}"/>
        <rect width="{total_width}" height="20" fill="url(#s)"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu \
Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
        <text aria-hidden="true" x="{int(label_x)}" y="150" fill="#010101" \
fill-opacity=".3" transform="scale(.1)" \
textLength="{label_width * 10 - 100}">{label_text}</text>
        <text x="{int(label_x)}" y="140" transform="scale(.1)" fill="#fff" \
textLength="{label_width * 10 - 100}">{label_text}</text>
        <text aria-hidden="true" x="{int(value_x)}" y="150" fill="#010101" \
fill-opacity=".3" transform="scale(.1)" \
textLength="{value_width * 10 - 100}">{value_text}</text>
        <text x="{int(value_x)}" y="140" transform="scale(.1)" fill="#fff" \
textLength="{value_width * 10 - 100}">{value_text}</text>
    </g>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Generated badge: {output_path} ({value_text})")


def get_complexity(file_path):
    """Calculates average cyclomatic complexity using radon."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        blocks = cc_visit(code)
        if not blocks:
            return 0

        total_cc = sum(b.complexity for b in blocks)
        return round(total_cc / len(blocks), 2)
    except Exception as e:
        print(f"DEBUG: Complexity calculation Error for {file_path}: {e}")
        return 0


def update_complexity(root):
    """Update complexity for each class using radon."""
    for cls in root.findall('.//class'):
        filename = cls.get('filename')
        if filename and filename.endswith('.py'):
            # Try to find the file: absolute, relative, or basename
            current_dir = os.getcwd()
            possible_paths = [
                filename,
                os.path.join(current_dir, filename),
                os.path.basename(filename)
            ]
            target_file = None
            for p in possible_paths:
                if os.path.exists(p):
                    target_file = p
                    break

            if target_file:
                # Calculate complexity
                cc = get_complexity(target_file)
                cls.set('complexity', str(cc))
                print(f"DEBUG: Updated complexity for {filename} -> {cc}")
            else:
                print(f"DEBUG: Could not resolve file for {filename}")
                cls.set('complexity', '0')


def transform_coverage(xml_file):
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found")
        sys.exit(1)

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Update complexity for each class using radon
        update_complexity(root)

        # Generate badge from root line-rate
        root_line_rate = root.get("line-rate", "0")
        generate_badge(root_line_rate)

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)

    packages_el = root.find('packages')
    if packages_el is None:
        print("No <packages> element found")
        sys.exit(1)

    # Collect all classes from all existing packages
    all_classes = []
    for pkg in packages_el.findall('package'):
        classes_el = pkg.find('classes')
        if classes_el is not None:
            all_classes.extend(classes_el.findall('class'))

    # Clear existing packages
    packages_el.clear()

    # Create new package per class
    for cls in all_classes:
        filename = cls.get('filename')
        pkg_name = filename if filename else "unknown"

        new_pkg = ET.SubElement(packages_el, 'package')
        new_pkg.set('name', pkg_name)

        for attr in ['line-rate', 'branch-rate', 'complexity']:
            val = cls.get(attr)
            if val is not None:
                new_pkg.set(attr, str(val))
            else:
                new_pkg.set(attr, '0.0')

        new_classes = ET.SubElement(new_pkg, 'classes')
        new_classes.append(cls)

    tree.write(xml_file, encoding='UTF-8', xml_declaration=True)


def generate_summary(xml_file):
    if not os.path.exists(xml_file):
        return

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        line_rate = float(root.get("line-rate", "0")) * 100
        branch_rate = float(root.get("branch-rate", "0")) * 100
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("📊 Code Coverage Report")
        print()
        print(f"Total Coverage: {line_rate:.2f}%")
        print(f"Branch Coverage: {branch_rate:.2f}%")
        print(f"Generated: {generated_at}")
        print()
        print("File | Coverage | Branches | Complexity")
        print("--- | --- | --- | ---")

        packages = root.find('packages')
        if packages is not None:
            for pkg in packages.findall('package'):
                name = pkg.get('name', 'unknown')
                l_rate = float(pkg.get('line-rate', '0')) * 100
                b_rate = float(pkg.get('branch-rate', '0')) * 100
                complexity = pkg.get('complexity', '0')
                print(f"{name} | {l_rate:.1f}% | {b_rate:.1f}% | {complexity}")

    except Exception as e:
        print(f"Error generating summary: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        xml_path = "coverage.xml"

    if "--summary" in sys.argv:
        generate_summary(xml_path)
    else:
        transform_coverage(xml_path)
