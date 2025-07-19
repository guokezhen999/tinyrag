from flask import Flask
from config import Config
from .services import AgentDialogService

agent_dialog_service = AgentDialogService()
print(">>> [INFO] AgentDialog Service initialized successfully.")

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)
    from . import routes
    app.register_blueprint(routes.bp)
    print(">>> [SUCCESS] Flask application created and configured.")
    return app