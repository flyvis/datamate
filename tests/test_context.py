import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from datamate.context import (
    check_size_on_init,
    context,
    delete_if_exists,
    enforce_config_match,
    get_check_size_on_init,
    get_default_scope,
    get_root_dir,
    get_scope,
    reset_scope,
    set_root_context,
    set_root_dir,
    set_scope,
    set_verbosity_level,
)


# Test root directory management
def test_root_dir_management(tmp_path):
    # Test default root dir
    assert get_root_dir() == Path(".")

    # Test setting root dir
    set_root_dir(tmp_path)
    assert get_root_dir() == tmp_path

    # Test setting None reverts to current directory
    set_root_dir(None)
    assert get_root_dir() == Path(".")


def test_root_context(tmp_path):
    original_root = get_root_dir()

    with set_root_context(tmp_path):
        assert get_root_dir() == tmp_path
        assert context.within_root_context.get() is True

    assert get_root_dir() == original_root
    assert context.within_root_context.get() is False


def test_delete_if_exists_context():
    assert context.delete_if_exists.get() is False

    with delete_if_exists():
        assert context.delete_if_exists.get() is True

    assert context.delete_if_exists.get() is False

    with delete_if_exists(False):
        assert context.delete_if_exists.get() is False


def test_enforce_config_match():
    assert context.enforce_config_match.get() is True

    enforce_config_match(False)
    assert context.enforce_config_match.get() is False

    enforce_config_match(True)
    assert context.enforce_config_match.get() is True


def test_check_size_on_init():
    assert get_check_size_on_init() is False

    check_size_on_init(True)
    assert get_check_size_on_init() is True

    check_size_on_init(False)
    assert get_check_size_on_init() is False


def test_verbosity_level():
    assert context.verbosity_level.get() == 1

    set_verbosity_level(0)
    assert context.verbosity_level.get() == 0

    set_verbosity_level(2)
    assert context.verbosity_level.get() == 2

    with pytest.raises(ValueError):
        set_verbosity_level(3)


def test_scope_management():
    # Test default scope
    default_scope = get_default_scope()
    assert isinstance(default_scope, dict)
    assert "Directory" in default_scope

    # Test setting custom scope
    custom_scope = {"CustomClass": type("CustomClass", (), {})}
    set_scope(custom_scope)
    assert get_scope() == custom_scope

    # Test resetting scope
    reset_scope()
    assert get_scope() == get_default_scope()

    # Test setting None scope returns default
    set_scope(None)
    assert get_scope() == get_default_scope()


@pytest.mark.asyncio
async def test_async_root_context(tmp_path):
    """Test that root context works correctly in async functions."""
    original_root = get_root_dir()

    async def nested_context_check():
        assert get_root_dir() == tmp_path
        assert context.within_root_context.get() is True

    with set_root_context(tmp_path):
        await nested_context_check()
        # Test concurrent tasks maintain correct context
        tasks = [nested_context_check() for _ in range(3)]
        await asyncio.gather(*tasks)

    assert get_root_dir() == original_root
    assert context.within_root_context.get() is False


@pytest.mark.asyncio
async def test_async_scope_isolation():
    """Test that scope remains isolated between different async tasks."""
    custom_scope = {"CustomClass": type("CustomClass", (), {})}

    async def task1():
        set_scope(custom_scope)
        await asyncio.sleep(0.1)  # Simulate some async work
        assert get_scope() == custom_scope

    async def task2():
        await asyncio.sleep(0.05)  # Start checking mid-way through task1
        assert get_scope() == get_default_scope()  # Should have default scope

    # Run tasks concurrently
    await asyncio.gather(task1(), task2())

    # Verify we're back to default scope
    assert get_scope() == get_default_scope()


@pytest_asyncio.fixture(autouse=True)
async def reset_context_vars():
    """Reset all context variables to their defaults before each test."""
    context.verbosity_level.set(1)
    context.check_size_on_init.set(False)
    context.enforce_config_match.set(True)
    yield


@pytest.mark.asyncio
async def test_async_context_vars():
    """Test all context variables maintain proper isolation in async contexts."""

    async def modify_contexts():
        # Store tokens to properly reset context
        v_token = context.verbosity_level.set(2)
        s_token = context.check_size_on_init.set(True)
        c_token = context.enforce_config_match.set(False)
        try:
            await asyncio.sleep(0.1)
            return (
                context.verbosity_level.get(),
                get_check_size_on_init(),
                context.enforce_config_match.get(),
            )
        finally:
            # Reset context using tokens
            context.verbosity_level.reset(v_token)
            context.check_size_on_init.reset(s_token)
            context.enforce_config_match.reset(c_token)

    async def check_default_contexts():
        await asyncio.sleep(0.05)
        return (
            context.verbosity_level.get(),
            get_check_size_on_init(),
            context.enforce_config_match.get(),
        )

    # Run concurrent tasks
    modified, default = await asyncio.gather(
        modify_contexts(), check_default_contexts()
    )

    # Modified task should see its changes
    assert modified == (2, True, False)

    # Default task should see original values
    assert default == (1, False, True)

    # After tasks complete, we should still have original values
    assert context.verbosity_level.get() == 1
    assert get_check_size_on_init() is False
    assert context.enforce_config_match.get() is True
