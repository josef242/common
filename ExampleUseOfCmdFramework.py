from CommandFramework import CommandFramework

# Example usage
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: str, b: str) -> str:
    try:
        return str(int(a) + int(b))
    except ValueError:
        return "Error: Please provide two valid integers."
    
def cmd_exit():
    print("Goodbye!")
    exit(0)

if __name__ == "__main__":
    print("Welcome to the Command Framework CLI!")
    print("Type '/help' to list available commands, '/usage <command>' for command usage, or '/exit' to quit.")
    
    # Create and set up the framework
    framework = CommandFramework("/")
    framework.add_command("greet", greet, "Greet a person by name")
    framework.add_command("add", add, "Add two numbers")
    framework.add_command("exit", cmd_exit, "Exit the program")
    
    while True:
        is_cmd, command, args = framework.get_user_command()
        if not command or not is_cmd:
            continue

        try:
            result = framework.execute(command, *args)
            print(result)
        except ValueError as e:
            print(f"Error: {str(e)}")
        except TypeError as e:
            print(f"Error: Invalid arguments. Use 'usage {command}' to see the correct usage.")