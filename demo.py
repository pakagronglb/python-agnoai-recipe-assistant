from setup_agent import recipe_agent
from agno.playground import Playground, serve_playground_app

agent = recipe_agent()

app = Playground(agents=[agent]).get_app()

if __name__ == '__main__':
    serve_playground_app('demo:app', reload=True)
