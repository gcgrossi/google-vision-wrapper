import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="google-vision-wrapper",
    version="0.0.2",
    description="Tiny Python wrapper for Google Vision API. Query google vision API and obtain information in pandas DataFrame in few lines of code.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/gcgrossi/google-vision-wrapper",
    author="Giulio Cornelio Grossi, Ph.D.",
    author_email="giulio.cornelio.grossi@gmail.com",
    license="MIT",
    classifiers=[],
    packages=["gvision"],
    install_requires=["google-cloud-vision>2","opencv-python","pandas","numpy"],
)