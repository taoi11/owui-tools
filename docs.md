Toolkits are defined in a single Python file, with a top level docstring with metadata and a `Tools` class.

### Example Top-Level Docstring[​](https://docs.openwebui.com/features/plugin/tools/development.html#example-top-level-docstring "Direct link to Example Top-Level Docstring")

```
"""title: String Inverseauthor: Your Nameauthor_url: https://website.comgit_url: https://github.com/username/string-reverse.gitdescription: This tool calculates the inverse of a stringrequired_open_webui_version: 0.4.0requirements: langchain-openai, langgraph, ollama, langchain_ollamaversion: 0.4.0licence: MIT""" 
```

### Tools Class[​](https://docs.openwebui.com/features/plugin/tools/development.html#tools-class "Direct link to Tools Class")

Tools have to be defined as methods within a class called `Tools`, with optional subclasses called `Valves` and `UserValves`, for example:

```
class Tools:    def __init__(self):        """Initialize the Tool."""        self.valves = self.Valves()    class Valves(BaseModel):        api_key: str = Field("", description="Your API key here")    def reverse_string(self, string: str) -> str:        """        Reverses the input string.        :param string: The string to reverse        """        # example usage of valves        if self.valves.api_key != "42":            return "Wrong API key"        return string[-1] 
```

### Type Hints[​](https://docs.openwebui.com/features/plugin/tools/development.html#type-hints "Direct link to Type Hints")

Each tool must have type hints for arguments. As of version Open WebUI version 0.4.3, the types may also be nested, such as `queries_and_docs: list[tuple[str, int]]`. Those type hints are used to generate the JSON schema that is sent to the model. Tools without type hints will work with a lot less consistency.

### Valves and UserValves - (optional, but HIGHLY encouraged)[​](https://docs.openwebui.com/features/plugin/tools/development.html#valves-and-uservalves---optional-but-highly-encouraged "Direct link to Valves and UserValves - (optional, but HIGHLY encouraged)")

Valves and UserValves are used to allow users to provide dynamic details such as an API key or a configuration option. These will create a fillable field or a bool switch in the GUI menu for the given function.

Valves are configurable by admins alone and UserValves are configurable by any users.

Commented example

```
from pydantic import BaseModel, Fieldclass Tools:   # Notice the current indentation: Valves and UserValves must be declared as    # attributes of a Tools, Filter or Pipe class. Here we take the    # example of a Tool.     class Valves(BaseModel):       # Valves and UserValves inherit from pydantic's BaseModel. This# enables complex use cases like model validators etc.       test_valve: int = Field(  # Notice the type hint: it is used to         # choose the kind of UI element to show the user (buttons,texts,etc).           default=4,           description="A valve controlling a numberical value"           # required=False,  # you can enforce fields using True        )       pass     # Note that this 'pass' helps for parsing and is recommended.   # UserValves are defined the same way.   class UserValves(BaseModel):       test_user_valve: bool = Field(            default=False, description="A user valve controlling a True/False (on/off) switch"       )       pass   def __init__(self):        self.valves = self.Valves()       # Because they are set by the admin, they are accessible directly       # upon code execution.       pass   # The  __user__ handling is the same for Filters, Tools and Functions.   def test_the_tool(self, message: str, __user__: dict):        """        This is a test tool. If the user asks you to test the tools, put any        string you want in the message argument.        :param message: Any string you want.        :return: The same string as input.        """        # Because UserValves are defined per user they are only available       # on use.       # Note that although __user__ is a dict, __user"]["valves"] is a       # UserValves object. Hence you can access values like that:       test_user_valve = __user__["valves"].test_user_valve       # Or:       test_user_valve = dict(__user__["valves"])["test_user_valve"]       # But this will return the default value instead of the actual value:       # test_user_valve = __user__["valves"]["test_user_valve"]  # Do not do that!        return message + f"\nThe user valve set value is: {test_user_valve}"       
```

### Optional Arguments[​](https://docs.openwebui.com/features/plugin/tools/development.html#optional-arguments "Direct link to Optional Arguments")

Below is a list of optional arguments your tools can depend on:

*   `__event_emitter__`: Emit events (see following section)
*   `__event_call__`: Same as event emitter but can be used for user interactions
*   `__user__`: A dictionary with user information. It also contains the `UserValves` object in `__user__["valves"]`.
*   `__metadata__`: Dictionary with chat metadata
*   `__messages__`: List of previous messages
*   `__files__`: Attached files
*   `__model__`: Model name

Just add them as argument to any method of your Tool class just like `__user__` in the example above.

### Event Emitters[​](https://docs.openwebui.com/features/plugin/tools/development.html#event-emitters "Direct link to Event Emitters")

Event Emitters are used to add additional information to the chat interface. Similarly to Filter Outlets, Event Emitters are capable of appending content to the chat. Unlike Filter Outlets, they are not capable of stripping information. Additionally, emitters can be activated at any stage during the Tool.

There are two different types of Event Emitters:

If the model seems to be unable to call the tool, make sure it is enabled (either via the Model page or via the `+` sign next to the chat input field). You can also turn the `Function Calling` argument of the `Advanced Params` section of the Model page from `Default` to `Native`.

#### Status[​](https://docs.openwebui.com/features/plugin/tools/development.html#status "Direct link to Status")

This is used to add statuses to a message while it is performing steps. These can be done at any stage during the Tool. These statuses appear right above the message content. These are very useful for Tools that delay the LLM response or process large amounts of information. This allows you to inform users what is being processed in real-time.

```
await __event_emitter__(            {                "type": "status", # We set the type here                "data": {"description": "Message that shows up in the chat", "done": False, "hidden": False},                 # Note done is False here indicating we are still emitting statuses            }        ) 
```

Example

```
async def test_function(         self, prompt: str, __user__: dict, __event_emitter__=None) -> str:        """        This is a demo        :param test: this is a test parameter        """        await __event_emitter__(            {                "type": "status", # We set the type here                "data": {"description": "Message that shows up in the chat", "done": False},                 # Note done is False here indicating we are still emitting statuses            }        )        # Do some other logic here        await __event_emitter__(            {                "type": "status",                "data": {"description": "Completed a task message", "done": True, "hidden": False},             # Note done is True here indicating we are done emitting statuses                # You can also set "hidden": True if you want to remove the status once the message is returned            }        )        except Exception as e:            await __event_emitter__(                {                    "type": "status",                    "data": {"description": f"An error occured: {e}", "done": True},                }            )            return f"Tell the user: {e}" 
```

#### Message[​](https://docs.openwebui.com/features/plugin/tools/development.html#message "Direct link to Message")

This type is used to append a message to the LLM at any stage in the Tool. This means that you can append messages, embed images, and even render web pages before, or after, or during the LLM response.

```
await __event_emitter__(                    {                        "type": "message", # We set the type here                        "data": {"content": "This message will be appended to the chat."},                        # Note that with message types we do NOT have to set a done condition                    }                ) 
```

Example

```
async def test_function(         self, prompt: str, __user__: dict, __event_emitter__=None) -> str:        """        This is a demo        :param test: this is a test parameter        """        await __event_emitter__(                    {                        "type": "message", # We set the type here                        "data": {"content": "This message will be appended to the chat."},                        # Note that with message types we do NOT have to set a done condition                    }                )        except Exception as e:            await __event_emitter__(                {                    "type": "status",                    "data": {"description": f"An error occured: {e}", "done": True},                }            )            return f"Tell the user: {e}" 
```

#### Citations[​](https://docs.openwebui.com/features/plugin/tools/development.html#citations "Direct link to Citations")

This type is used to provide citations or references in the chat. You can utilize it to specify the content, the source, and any relevant metadata. Below is an example of how to emit a citation event:

```
await __event_emitter__(    {        "type": "citation",        "data": {            "document": [content],            "metadata": [                {                    "date_accessed": datetime.now().isoformat(),                    "source": title,                }            ],            "source": {"name": title, "url": url},        },    }) 
```

If you are sending multiple citations, you can iterate over citations and call the emitter multiple times. When implementing custom citations, ensure that you set `self.citation = False` in your `Tools` class `__init__` method. Otherwise, the built-in citations will override the ones you have pushed in. For example:

```
def __init__(self):    self.citation = False 
```

Warning: if you set `self.citation = True`, this will replace any custom citations you send with the automatically generated return citation. By disabling it, you can fully manage your own citation references.

Example

```
class Tools:    class UserValves(BaseModel):        test: bool = Field(            default=True, description="test"        )    def __init__(self):        self.citation = Falseasync def test_function(         self, prompt: str, __user__: dict, __event_emitter__=None) -> str:        """        This is a demo that just creates a citation        :param test: this is a test parameter        """        await __event_emitter__(            {                "type": "citation",                "data": {                    "document": ["This message will be appended to the chat as a citation when clicked into"],                    "metadata": [                        {                            "date_accessed": datetime.now().isoformat(),                            "source": title,                        }                    ],                    "source": {"name": "Title of the content", "url": "http://link-to-citation"},                },            }        ) 
```

External packages[​](https://docs.openwebui.com/features/plugin/tools/development.html#external-packages "Direct link to External packages")
---------------------------------------------------------------------------------------------------------------------------------------------

In the Tools definition metadata you can specify custom packages. When you click `Save` the line will be parsed and `pip install` will be run on all requirements at once.

Keep in mind that as pip is used in the same process as Open WebUI, the UI will be completely unresponsive during the installation.

No measures are taken to handle package conflicts with Open WebUI's requirements. That means that specifying requirements can break Open WebUI if you're not careful. You might be able to work around this by specifying `open-webui` itself as a requirement.

Example

```
"""title: myToolNameauthor: myNamefunding_url: [any link here will be shown behind a `Heart` button for users to show their support to you]version: 1.0.0# the version is displayed in the UI to help users keep track of updates.license: GPLv3description: [recommended]requirements: package1>=2.7.0,package2,package3""" 
```