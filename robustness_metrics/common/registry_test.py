# coding=utf-8
# Copyright 2020 The Robustness Metrics Authors.
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
"""Tests for the registry."""
from absl.testing import absltest
from absl.testing import parameterized
import robustness_metrics as rm


class Base:
  pass  # The base class used in the tests.


class RegistryTest(parameterized.TestCase, absltest.TestCase):

  def test_explicit_registry(self):
    registry = rm.common.registry.Registry(Base)

    class Child(Base):
      pass

    registry.register_subclass("child", Child)
    self.assertEqual(registry.get("child"), Child)

  def test_registry_using_a_decorator(self):
    registry = rm.common.registry.Registry(Base)

    @registry.register("child")
    class Child(Base):
      pass

    self.assertEqual(registry.get("child"), Child)

  def test_that_exception_is_raised_on_name_reuse(self):
    registry = rm.common.registry.Registry(Base)

    @registry.register("child")
    class Child(Base):
      pass

    expected_error_message = (
        "You are registering a subclass with name 'child', which has been "
        "already taken by (.*)")
    with self.assertRaisesRegex(ValueError, expected_error_message):
      registry.register_subclass("child", Child)

  def test_that_exception_is_raised_on_unknown_name(self):
    registry = rm.common.registry.Registry(Base)
    expected_error_message = (
        "Unknown subclass 'foobar', registered names: '(.*)'")
    with self.assertRaisesRegex(KeyError, expected_error_message):
      registry.get("foobar")

  def test_registered_subclasses_retrieval(self):
    registry = rm.common.registry.Registry(Base)

    class Child(Base):
      pass

    registry.register_subclass("child1", Child)
    registry.register_subclass("child2", Child)
    registered_subclasses = registry.get_registered_subclasses()
    self.assertCountEqual(["child1", "child2"], registered_subclasses)

  def test_that_instance_initialization(self):
    registry = rm.common.registry.Registry(Base)

    class SubclassWithInit(Base):

      def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar

    registry.register_subclass("with_init", SubclassWithInit)
    self.assertEqual(registry.get("with_init"), SubclassWithInit)
    instance = registry.get_instance("with_init(foo=123,bar='barz')")
    assert isinstance(instance, SubclassWithInit)
    self.assertEqual(instance.foo, 123)
    self.assertEqual(instance.bar, "barz")

    instance = registry.get_instance("with_init(foo=123)", bar="barz")
    assert isinstance(instance, SubclassWithInit)
    self.assertEqual(instance.foo, 123)
    self.assertEqual(instance.bar, "barz")

    expected_error_message = "(.*) is not a valid Python code."
    with self.assertRaisesRegex(ValueError, expected_error_message):
      # Right parenthesis is missing in the spec.
      registry.get_instance("with_init(foo=123,bar='barz'")

  @parameterized.parameters([
      ("foo(a=1,c='bar',b=2)", """foo(b=2,a=1,c="bar")"""),
      ("bar()", """bar"""),
  ])
  def test_standardization(self, spec_1, spec_2):
    spec_1_std = rm.common.registry.standardize_spec(spec_1)
    spec_2_std = rm.common.registry.standardize_spec(spec_2)
    self.assertEqual(spec_1_std, spec_2_std)
    self.assertEqual(rm.common.registry.parse_name_and_kwargs(spec_1),
                     rm.common.registry.parse_name_and_kwargs(spec_1_std))
    self.assertEqual(rm.common.registry.parse_name_and_kwargs(spec_2),
                     rm.common.registry.parse_name_and_kwargs(spec_2_std))


if __name__ == "__main__":
  absltest.main()
