""" Module utils """

from aws_cdk import core


# TODO: this method should probably be put in a FR CDK library as it will be used quite often
def consolidate_context(stack: core.Stack, target: str) -> dict:
    """ Consolidate the stack's context based on target as a single dict.
    Note: this is not the full context but just common + target.
    The target must exist as a key in common.
    Args:
        stack: The CDK stack object to work with.
        target: The target to work with.

    Returns:
        The consolidated common+target as a simple dictionary.
    """
    # From the context, load common configuration + target' configuration
    config = stack.node.try_get_context("common")
    config.update(stack.node.try_get_context(target))

    return config
