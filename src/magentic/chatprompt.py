import inspect
from functools import update_wrapper
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    ParamSpec,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

from magentic.backend import get_chat_model
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import Message
from magentic.typing import split_union_type

P = ParamSpec("P")
# TODO: Make `R` type Union of all possible return types except FunctionCall ?
# Then `R | FunctionCall[FuncR]` will separate FunctionCall from other return types.
# Can then use `FuncR` to make typechecker check `functions` argument to `chatprompt`
# `Not` type would solve this - https://github.com/python/typing/issues/801
R = TypeVar("R")


def escape_braces(text: str) -> str:
    """Escape curly braces in a string.

    This allows curly braces to be used in a string template without being interpreted
    as format specifiers.
    """
    return text.replace("{", "{{").replace("}", "}}")


class BaseChatPromptFunction(Generic[P, R]):
    """Base class for an LLM chat prompt template that is directly callable to query the LLM."""

    def __init__(
        self,
        messages: Sequence[Message[Any]],
        parameters: Sequence[inspect.Parameter],
        return_type: type[R],
        functions: list[Callable[..., Any]] | None = None,
        stop: list[str] | None = None,
        model: ChatModel | None = None,
    ):
        self._signature = inspect.Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._messages = messages
        self._functions = functions or []
        self._stop = stop
        self._model = model

        self._return_types = list(split_union_type(return_type))

    @property
    def functions(self) -> list[Callable[..., Any]]:
        return self._functions.copy()

    @property
    def model(self) -> ChatModel:
        return self._model or get_chat_model()

    @property
    def return_types(self) -> list[type[R]]:
        return self._return_types.copy()

    def format(self, *args: P.args, **kwargs: P.kwargs) -> list[Message[Any]]:
        """Format the message templates with the given arguments."""
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return [
            message_template.format(**bound_args.arguments)
            for message_template in self._messages
        ]


class ChatPromptFunction(BaseChatPromptFunction[P, R], Generic[P, R]):
    """An LLM chat prompt template that is directly callable to query the LLM."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Query the LLM with the formatted chat prompt template."""
        message = self.model.complete(
            messages=self.format(*args, **kwargs),
            functions=self._functions,
            output_types=self._return_types,
            stop=self._stop,
        )
        return message.content


class AsyncChatPromptFunction(BaseChatPromptFunction[P, R], Generic[P, R]):
    """Async version of `ChatPromptFunction`."""

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Asynchronously query the LLM with the formatted chat prompt template."""
        message = await self.model.acomplete(
            messages=self.format(*args, **kwargs),
            functions=self._functions,
            output_types=self._return_types,
            stop=self._stop,
        )
        return message.content


class ChatPromptDecorator(Protocol):
    """Protocol for a decorator that returns a `ChatPromptFunction`.

    This allows finer-grain type annotation of the `chatprompt` function
    See https://github.com/microsoft/pyright/issues/5014#issuecomment-1523778421
    """

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Awaitable[R]]
    ) -> AsyncChatPromptFunction[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, R]) -> ChatPromptFunction[P, R]: ...


def chatprompt(
    *messages: Message[Any],
    functions: list[Callable[..., Any]] | None = None,
    stop: list[str] | None = None,
    model: ChatModel | None = None,
) -> "ChatPromptDecorator":
    """Convert a function into an LLM chat prompt template."""

    def decorator(
        func: Callable[P, Awaitable[R]] | Callable[P, R],
    ) -> Union[AsyncChatPromptFunction[P, R], ChatPromptFunction[P, R]]:
        func_signature = inspect.signature(func)
        params = {
            "messages": messages,
            "parameters": list(func_signature.parameters.values()),
            "return_type": func_signature.return_annotation,
            "functions": functions,
            "stop": stop,
            "model": model,
        }

        if inspect.iscoroutinefunction(func):
            async_prompt_function = AsyncChatPromptFunction(**params)
            return cast(
                AsyncChatPromptFunction[P, R],
                update_wrapper(async_prompt_function, func),
            )

        prompt_function = ChatPromptFunction(**params)
        return cast(ChatPromptFunction[P, R], update_wrapper(prompt_function, func))

    return cast(ChatPromptDecorator, decorator)
