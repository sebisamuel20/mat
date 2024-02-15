import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.1"

REPO_NAME = "Maternity Health Risk"
AUTHOR_USER_NAME = "sebisamuel20"
SRC_REPO = 'new_proj'
AUTHOR_EMAIL = "sebisamuel20@gmail.com"

setuptools.setup(
    name= SRC_REPO,
    version= __version__,
    author= AUTHOR_USER_NAME,
    author_email= AUTHOR_EMAIL,
    description= "A small package to detect maternity risk",
    long_description= long_description,
    long_description_content = "text/markdown",
    package_dir={"": "src"},
    packages= setuptools.find_packages(where = "src")
    )