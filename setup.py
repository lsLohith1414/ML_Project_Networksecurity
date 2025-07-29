from setuptools import setup, find_packages
from typing import List


HIPN_E_DOT = "-e ."


def get_requirements(file_path:str)->List[str]:

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [line.strip() for line in requirements]

        if HIPN_E_DOT in requirements:
            requirements.remove(HIPN_E_DOT)

    return requirements        




setup(
    name="Network security project",
    version="0.0.1",
    description="This is an Data science porject",
    author="Lohith H S",
    author_email="lohithls14@gmail.com",
    maintainer_email="lohithls14@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)

install_requires = get_requirements("requirements.txt")
print(install_requires)