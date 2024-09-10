import os
from flask import Flask
from flask_cors import CORS
import torch
from app.llm import LLM

def create_app():
    print("Creating App...")
    app = Flask(__name__)
    CORS(app)
    
    torch.random.manual_seed(12)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        llm = LLM()
    
    # Register Generate Blueprint
    from app.routes.generate import generate_bp
    app.register_blueprint(generate_bp)
    
    # Register Generate Dataset Blueprint
    from app.routes.generate_dataset import generate_dataset_bp
    app.register_blueprint(generate_dataset_bp)

    return app
