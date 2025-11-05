import functools

import inspect
from typing import Any, Callable, Dict, Tuple, Optional


class HookContext:
    def __init__(self):
        self._patches: Dict[Tuple[Any, str], Any] = {}

    def hook(
        self,
        obj: Any,
        name: str,
        before: Optional[Callable[[Dict, Dict], Dict]] = None,
        after: Optional[
            Callable[[Dict, Optional[Any], Optional[BaseException], Dict], None]
        ] = None,
    ):
        value = getattr(
            obj,
            name,
        )

        if self._patches.get((obj, name), None) is not None:
            raise ValueError(f"Already hooked: {obj}.{name}")

        self._patches[(obj, name)] = value

        context = {}
        before = functools.partial(before, context) if before else None
        after = functools.partial(after, context) if after else None

        def _binder(
            *args,
            **kwargs,
        ) -> Dict[str, Any]:
            sig = inspect.signature(value)
            args = sig.bind(*args, **kwargs)
            args.apply_defaults()
            return args.arguments

        def _hooker(
            *args,
            **kwargs,
        ):
            kwargs = _binder(
                *args,
                **kwargs,
            )
            if before is not None:
                before(
                    kwargs,
                )
            retval, error = None, None
            try:
                retval = value(**kwargs)
            except BaseException as e:
                error = e
                raise error
            finally:
                if after is not None:
                    after(kwargs, retval, error)

        setattr(
            obj,
            name,
            _hooker,
        )

    def unhook(
        self,
        obj: Any,
        name: str,
    ) -> None:
        if self._patches.get((obj, name), None) is None:
            raise ValueError(f"Not hooked: {obj}.{name}")

        setattr(
            obj,
            name,
            self._patches.pop((obj, name)),
        )

    def unhook_all(
        self,
    ) -> None:
        for (obj, name), value in self._patches.items():
            setattr(
                obj,
                name,
                value,
            )
        self._patches.clear()
