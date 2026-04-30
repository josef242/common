import inspect
from typing import Callable, Dict, List, Any
import time

class Command:
    def __init__(self, name: str, handler: Callable, description: str):
        self.name = name
        self.handler = handler
        self.description = description

class CommandFramework:
    def __init__(self, prefix: str):
        self.commands: Dict[str, Command] = {}

        # Register the prefix for commands
        self.prefix = prefix
        
        # Register built-in commands
        self.add_command("help", self.cmd_help, "List all available commands")
        self.add_command("usage", self.cmd_usage, "Show usage for a specific command")

    # This function will be called to add a new command
    def add_command(self, name: str, handler: Callable, description: str):
        self.commands[name.lower()] = Command(name, handler, description)

    # This function will be called to execute a command
    def execute(self, command_name: str, *args) -> Any:
        command = self.commands.get(command_name.lower())
        if not command:
            return f"Command '{command_name}' not found."
        try:
            return command.handler(*args)
        except TypeError:
            signature = self.get_command_signature(command)
            return f"Invalid arguments for '{command_name}'. Usage: {self.prefix}{command.name} {signature}"

    # This function will be called to get the signature of a command
    def get_command_signature(self, command: Command) -> str:
        params = inspect.signature(command.handler).parameters
        return ' '.join(f'<{param}>' for param in params)

    # This function will be called when the user types 'help' - it will list all available commands
    def cmd_help(self) -> str:
        # Calculate the maximum widths for each column
        name_width = max(len(cmd.name) for cmd in self.commands.values())
        signature_width = max(len(self.get_command_signature(cmd)) for cmd in self.commands.values())
        desc_width = max(len(cmd.description) for cmd in self.commands.values())

        # Create the table header
        header = f"{'Command':<{name_width}} {'Arguments':<{signature_width}} {'Description':<{desc_width}}"
        separator = '-' * (name_width + signature_width + desc_width + 4)

        # Create the table rows
        rows = [
            f"{self.prefix}{cmd.name:<{name_width}} {self.get_command_signature(cmd):<{signature_width}} {cmd.description:<{desc_width}}"
            for cmd in self.commands.values()
        ]

        # Combine all parts
        return "Available commands:\n" + header + '\n' + separator + '\n' + '\n'.join(rows)

    # This function will be called when the user types 'usage <command>' - it will show the usage of the command
    def cmd_usage(self, command_name: str) -> str:
        command = self.commands.get(command_name.lower())
        if not command:
            return f"Command '{command_name}' not found."
        
        signature = self.get_command_signature(command)
        return f"Usage: {self.prefix}{command.name} {signature}\nDescription: {command.description}"
    
    # This function will be called in a loop to get user input and execute commands
    def do_user_command(self, opt_prompt= "\n>> "):
        # sleep for 1 second to simulate a delay (optional)
        time.sleep(1)
        user_input = input(f"{opt_prompt}").strip().split()
        if not user_input:
            return False, ""
    
        # Check to see if the input starts with the command prefix
        is_cmd = user_input[0].startswith(self.prefix)

        # If it doesn't, it's not a command - return the input as is
        if not is_cmd:
            return False, user_input

        # Be sure to skip the prefix when extracting the command and arguments
        command = user_input[0][len(self.prefix):]
        args = user_input[1:]

        result = self.execute(command, *args)
        print(result)

        return is_cmd, command