from setuptools import find_packages, setup

with open("deps/requirements.txt") as f:
    install_requires = f.read()


if __name__ == "__main__":
    setup(
        name="scale_rl",
        version="0.1.0",
        url="https://github.com/SonyResearch/simba",
        license="Apache License 2.0",
        install_requires=install_requires,
        packages=find_packages(),
        python_requires=">=3.9.0",
        zip_safe=True,
    )
