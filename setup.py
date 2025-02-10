import os
import shutil
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

def run_post_install_script():
    """Executes install_plink.py after installation."""
    script_path = os.path.join(os.path.dirname(__file__), "chasm", "install_plink.py")

    if os.path.exists(script_path):
        print(f"Running post-install script: {script_path}")
        try:
            subprocess.run(["python", script_path], check=True)  # More robust execution
        except subprocess.CalledProcessError as e:
            print(f"Post-install script failed: {e}")
    else:
        print(f"ERROR: {script_path} not found!")

def cleanup_build():
    """Removes the build directory after installation to prevent clutter."""
    build_path = os.path.join(os.path.dirname(__file__), "build")
    if os.path.exists(build_path):
        print(f"Cleaning up build directory: {build_path}")
        shutil.rmtree(build_path)  # Removes the build folder

class CustomInstallCommand(install):
    """Custom install command to run post-install script and cleanup build dir."""
    def run(self):
        install.run(self)  # Run normal install
        run_post_install_script()
        cleanup_build()

class CustomDevelopCommand(develop):
    """Custom develop command to run post-install script and cleanup build dir."""
    def run(self):
        develop.run(self)  # Run normal develop install
        run_post_install_script()
        cleanup_build()

class CustomBdistWheelCommand(_bdist_wheel):
    """Custom bdist_wheel command to ensure cleanup after wheel build."""
    def run(self):
        _bdist_wheel.run(self)  # Run normal bdist_wheel
        cleanup_build()

setup(
    name='chasm',
    version='0.1',  # Ensure version is specified
    description="A package which gives a better representation of population representation",
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,  # Ensures execution during `pip install -e .`
        'bdist_wheel': CustomBdistWheelCommand,  # Ensures cleanup after wheel build
    },
)