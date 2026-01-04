import os


def pytest_sessionfinish(session, exitstatus):
    """
    Hook to run after the entire test session is finished.
    Automatically updates the coverage badge if coverage.xml exists.
    """
    xml_file = "coverage.xml"
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
