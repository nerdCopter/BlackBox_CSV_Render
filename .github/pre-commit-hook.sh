#!/bin/bash
# Pre-commit hook to automatically format Rust code and run clippy
# Install this by copying to .git/hooks/pre-commit and making it executable

echo "🔍 Running pre-commit checks..."

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ cargo not found. Please install Rust toolchain."
    exit 1
fi

# Run cargo fmt to format code
echo "📝 Formatting code with cargo fmt..."
if ! cargo fmt --all; then
    echo "❌ Code formatting failed!"
    exit 1
fi

# Check if formatting is compliant
echo "✅ Checking formatting compliance..."
if ! cargo fmt --all -- --check; then
    echo "❌ Code is not properly formatted after running cargo fmt!"
    echo "Please run 'cargo fmt --all' and try again."
    exit 1
fi

# Run clippy for linting
echo "🔍 Running clippy checks..."
if ! cargo clippy --all-targets --all-features -- -D warnings; then
    echo "❌ Clippy found issues that must be fixed!"
    exit 1
fi

echo "✅ All pre-commit checks passed!"
exit 0
