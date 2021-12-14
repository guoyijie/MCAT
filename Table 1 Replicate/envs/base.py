from abc import ABC, abstractmethod
import collections
import gym


class EnvBinarySuccessMixin(ABC):
    """Adds binary success metric to environment."""

    @abstractmethod
    def is_success(self):
         """Returns True is current state indicates success, False otherwise"""
         pass

