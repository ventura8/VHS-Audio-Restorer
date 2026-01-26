import os
import sys
import xml.etree.ElementTree as ET


def pytest_sessionfinish(session, exitstatus):
    """
    Hook to run after the entire test session is finished.
    Checks per-file coverage minimums and updates the coverage badge.
    """
    xml_file = "coverage.xml"
    
    # Update badge first
    if os.path.exists(xml_file):
        print("\nUpdating coverage badge...")
        try:
            # Import here to avoid E402 and mypy attribute errors
            from tests.transform_coverage import transform_coverage
            transform_coverage(xml_file)
            print("Coverage badge updated successfully.")
        except Exception as e:
            print(f"Failed to update coverage badge: {e}")
    else:
        print("\nNo coverage.xml found, skipping badge update.")
        return
    
    # Check per-file coverage from coverage.xml
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        min_coverage = 90.0
        files_below_threshold = []
        
        # Parse XML to get per-file coverage
        # Structure: coverage/packages/package (each represents a file)
        for package in root.findall('packages/package'):
            filename = package.get('name', '')
            line_rate = float(package.get('line-rate', '1')) * 100
            
            # Only check source files, not tests
            if filename and not filename.startswith('test') and line_rate < min_coverage:
                files_below_threshold.append((filename, line_rate))
        
        if files_below_threshold:
            print("\n" + "=" * 80)
            print("ERROR: Files below 90% coverage threshold:")
            print("=" * 80)
            for filename, coverage in sorted(files_below_threshold):
                print(f"  {filename:<50} {coverage:.2f}%")
            print("=" * 80)
            session.exitstatus = 1
    except Exception as e:
        print(f"Warning: Could not verify per-file coverage: {e}")


