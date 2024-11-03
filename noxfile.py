import nox
import sys

# sys.path.append("../")
LINT_DIR = ["symmetry_breaking_measure", "test"]


@nox.session(python="3.7", venv_backend="conda", reuse_venv=True)
def pytest_coverage(session):
    session.install("pytest")
    session.install("coverage")
    session.install("numpy")
    session.install("scipy")
    session.install("diffpy.structure")
    session.run("conda", "install", "bg-mpl-stylesheets")
    session.install("matplotlib")
    session.install("pymatgen")
    session.run("coverage", "run", "-m", "pytest", "test")
    session.run("coverage", "report", "-m")
    session.run("coverage", "html")


@nox.session(python="3.7")
def black(session):
    session.install("black")
    session.run("black", *LINT_DIR)


@nox.session(python="3.7")
def isort(session):
    session.install("isort")
    session.run("isort", *LINT_DIR, "--skip", "__init__.py")
