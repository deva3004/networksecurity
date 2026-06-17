# This file makes the project installable as a package via `pip install -e .`,
# which allows importing from `networksecurity` across any script or notebook
# without relying on sys.path hacks or relative imports.
import logging
from setuptools import find_packages, setup
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)

HYPHEN_E_DOT = "-e ."

def get_requirements() -> List[str]:
    try:
        requirements = []
        with open("requirements.txt") as f:
            requirements = f.readlines()
            requirements = [req.strip() for req in requirements]
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
        logging.info("requirements.txt loaded successfully: %s packages found", len(requirements))
        return requirements
    except FileNotFoundError:
        logging.error("requirements.txt not found — returning empty requirements list")
        return []
    except Exception as e:
        logging.error("Failed to read requirements.txt: %s", e)
        raise

try:
    setup(
        name="networksecurity",
        version="0.0.1",
        author="Devashish",
        author_email="tripathidevashish@gmail.com",
        packages=find_packages(),
        install_requires=get_requirements(),
    )
    logging.info("Package setup completed successfully")
except Exception as e:
    logging.error("Package setup failed: %s", e)
    raise
 