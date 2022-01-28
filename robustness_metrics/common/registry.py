# coding=utf-8
# Copyright 2022 The Robustness Metrics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A registry for registering subclasses by name.

Typical usage:
```
registry = Registry(AbstractParentClass)


@registry.register("child")
class Child(AbstractParentClass): pass

assert registry.get("child") == Child
```
"""
import ast
from typing import Any, List, Dict, Text, Tuple


def _get_name(expr):
  if isinstance(expr, ast.Call):
    return expr.func.id
  elif isinstance(expr, ast.Name):
    return expr.id
  else:
    raise ValueError(
        f"Could not parse function name from expression: {expr!r}.")


def parse_name_and_kwargs(code: str) -> Tuple[str, List[Any], Dict[str, Any]]:
  """Parse a Python function call into the function name and arguments.

  If just a function name is provided without arguments, the dictionary will be
  returned empty.

  For example, given the input string "foo(a=1, b='1')", the result will be
  the tuple `("foo", {a=1, b="1"})`. Similarly, "foo" results in `("foo", {})`.

  Args:
    code: Python string with a function call.

  Returns:
    A tuple of the name, list holding the positional arguments, and a
    string-keyed dictionary holding the arguments.
  Raises:
    ValueError: If the code is malformed.
  """
  try:
    expr = ast.parse(code, mode="eval").body  # pytype: disable=attribute-error
  except SyntaxError:
    raise ValueError(f"{code!r} is not a valid Python code.")
  name = _get_name(expr)

  # Simple case without arguments.
  if isinstance(expr, ast.Name):
    return name, [], {}
  else:
    assert isinstance(expr, ast.Call)
    args = [ast.literal_eval(x) for x in expr.args]
    kwargs = {kv.arg: ast.literal_eval(kv.value) for kv in expr.keywords}
    return name, args, kwargs


def standardize_spec(code: str) -> str:
  """Standardize a spec used for creating instances.

  When we specify an instance the specs "foo(a=1,b=2)" and "foo(b=2,a=1)" yield
  the same instance, but the strings themselves are different. This function
  computes a canonical form that is the same for all specs that have the same
  kwargs.

  Args:
    code: An instance spec, i.e., a Python string with a function call.
  Returns:
    A standardized spec.
  """
  name, args, kwargs = parse_name_and_kwargs(code)
  if args:
    raise ValueError("Only kwargs are allowed.")
  sorted_kwargs = sorted(kwargs.items())
  sorted_kwargs_str = ",".join(
      f"{kwarg}={value!r}" for kwarg, value in sorted_kwargs)
  if sorted_kwargs_str:
    return f"{name}({sorted_kwargs_str})"
  else:
    return name


class Registry:
  """Registers sub-classes of a given parent class by name."""

  def __init__(self, parent_class):
    """Initializes the object.

    Args:
      parent_class: The type whose descendants you want to register.
    """
    self._parent_class = parent_class
    self._registered_subclasses = {}

  def register_subclass(self, name: Text, subclass):
    """Register the subclass under the given name.

    Args:
      name: The name under which the subclass will be registered.
      subclass: A sub-class of `parent_class`.
    Raises:
      ValueError: If a class has been already registered under that name.
    """
    if not issubclass(subclass, self._parent_class):
      raise ValueError(f"You can not register {subclass}, "
                       f"as it does not subclass {self._parent_class}")
    if name in self._registered_subclasses:
      existing_class = self._registered_subclasses[name]
      raise ValueError(f"You are registering a subclass with name {name!r},"
                       f" which has been already taken by {existing_class}")
    else:
      self._registered_subclasses[name] = subclass

  def register(self, name: Text):
    """Creates a decorator to register the decorated class under the given name.

    Args:
      name: The name under which you want to register the decorated class.
    Returns:
      A decorator that registers the decorated class under name `name`.
    """
    def decorator(subclass):
      self.register_subclass(name, subclass)
      return subclass

    return decorator

  def get(self, name: Text):
    """Get the class registered under the given name.

    Args:
      name: The name of the class.
    Returns:
      The class registered under the given name.
    Raises:
      KeyError: If no class has been registered under that name.
    """
    try:
      return self._registered_subclasses[name]
    except KeyError:
      known_subclasses = ",".join(self._registered_subclasses)
      raise KeyError(
          f"Unknown subclass {name!r}, registered names: {known_subclasses!r}")

  def get_instance(self, spec: Text, **extra_kwargs):
    """Returns an instance of a registered subclass.

    The give spec should specify the name of the class and the arguments
    in parenthesis, i.e.,

      "foo(arg=1, foo='bar', baz='3', param=1.2)"

    Note that positional arguments are not supported.

    Args:
      spec: The spec identifying the name of the class and the arguments that
        should be used to construct the instance.
      **extra_kwargs: Will be passed to the initializer of the instance in
        addition to those parsed form the spec.
    Returns:
      An instance of the the provided class.
    Raises:
      KeyError: If no class has been registered under that name.
      ValueError: If the spec is malformed.
    """
    name, args, kwargs = parse_name_and_kwargs(spec)
    if args:
      raise ValueError("Only kwargs are allowed.")
    kwargs.update(extra_kwargs)
    return self.get(name)(**kwargs)

  def get_registered_subclasses(self):
    """Get the names of all registered subclasses."""
    return list(self._registered_subclasses.keys())

