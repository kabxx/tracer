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
        before: Optional[Callable[[Dict, Dict], None]] = None,
        after: Optional[
            Callable[[Dict, Dict, Optional[Any], Optional[BaseException]], None]
        ] = None,
    ):
        value = getattr(
            obj,
            name,
        )

        if self._patches.get((obj, name), None) is not None:
            raise ValueError(f"Already hooked: {obj}.{name}")

        self._patches[(obj, name)] = value

        def _binder(
            *args,
            **kwds,
        ) -> Dict[str, Any]:
            sig = inspect.signature(value)
            args = sig.bind(*args, **kwds)
            args.apply_defaults()
            return args.arguments

        def _hooker(
            *args,
            **kwds,
        ):
            context = {}

            kwds = _binder(
                *args,
                **kwds,
            )
            if before is not None:
                before(
                    context,
                    kwds,
                )
            retval, error = None, None
            try:
                retval = value(**kwds)
            except BaseException as e:
                error = e
                raise error
            finally:
                if after is not None:
                    after(context, kwds, retval, error)

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
