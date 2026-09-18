from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def create_app():
    app = Flask(__name__, template_folder="views/templates", static_folder="views/static")

    limiter = Limiter(get_remote_address, app=app, default_limits=["20 per hour", "5 per minute"])

    from app.controllers.simulation_controller import simulation_bp
    app.register_blueprint(simulation_bp)

    return app