import setuptools

setuptools.setup(
    name="molecular_magic",
    version="2.0.1",
    packages=["molmagic"],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "magic=molmagic.cli:main",
            "magic-generalize=model_analysis.top10_errors:cli_tool",
            "magic-inference=model_analysis.inference:cli_tool"
        ],
    },
)
