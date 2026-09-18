from app import create_app
import os

app = create_app()
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "changez-moi-en-dev-uniquement")

if __name__ == "__main__":
    app.run(debug=False)  # debug=True uniquement en local