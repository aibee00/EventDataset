"""
一个类装饰器的综合类为各种类模块提供注册功能
"""
from dataset.validation_benchmark.events.events import BaseEvent

class Registry:
    registry_dict = {}
    mapping = {
        "event_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_event(cls, name):
        r"""Register a event to registry with key 'name'

        Args:
            name: Key with which the event will be registered.

        Usage:

            from lavis.common.registry import registry
        """

        def wrap(event_cls):
            assert issubclass(
                event_cls, BaseEvent
            ), "All events must inherit BaseEvent class"
            if name in cls.mapping["event_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["event_name_mapping"][name]
                    )
                )
            cls.mapping["event_name_mapping"][name] = event_cls
            return event_cls

        return wrap
    
    @classmethod
    def get_event_class(cls, name):
        return cls.mapping["event_name_mapping"].get(name, None)
    

registry = Registry()
