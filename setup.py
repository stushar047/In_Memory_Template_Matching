from setuptools import setup, find_packages

setup(
    name="in_memory_template_matching",
    version="0.1.0",
    description="In-memory template matching with approximated PCC computation leveraging memristive systems",
    author="Sree Nirmillo Biswash Tushar",
    url="https://github.com/stushar047/In_Memory_Template_Matching",
    packages=find_packages(),
    py_modules=[],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "opencv-python",
        "Pillow",
    ],
    python_requires=">=3.9",
)
