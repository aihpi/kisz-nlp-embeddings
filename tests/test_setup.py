import pytest

import subprocess
import yaml

@pytest.mark.env
class TestEnvironment:
    @pytest.fixture(scope='class')
    def yaml(self, file_path: str='envs/environment.yml'):
        # Read and parse the YAML file
        with open(file_path, 'r', encoding='utf8') as yml_file:
            yml_data = yaml.safe_load(yml_file)
            dependencies = yml_data.get("dependencies", [])
        yield dependencies

    @pytest.fixture(scope='class')
    def req_py_version(self, yaml):
        for dep in yaml:
            if isinstance(dep, str):
                package_info = dep.split('=')
                if package_info[0] == 'python':
                    return package_info[1]
            
    @pytest.fixture(scope='class')
    def req_packages(self, yaml):
        required_packages = {}
        for dep in yaml:
            if isinstance(dep, str):
                package_info = dep.split('=')
                if package_info[0] != 'python':
                    required_packages[package_info[0]] = package_info[1]
            return required_packages

    @pytest.fixture(scope='class')
    def current_env(self):
        cmd = f"conda list"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        lines = stdout.split('\n')
        installed_packages = {}

        for line in lines[3:-1]:
            columns = line.split()
            installed_packages[columns[0]] = columns[1]

        return installed_packages
    
    @pytest.fixture(scope='class')
    def py_version(self, current_env):
        for name, version in current_env.items():
            if name == 'python':
                parts = version.split('.')
                return '.'.join(parts[:2])
            
    @pytest.fixture(scope='class')
    def packages(self, current_env):
        for name, version in current_env.items():
            parts = version.split('.')
            current_env[name] = '.'.join(parts[:2])
        return current_env

    def test_python_version(self, req_py_version, py_version):
        assert req_py_version == py_version

    def test_missing_packages(self, req_packages, packages):
        missing_packages = [name for name in req_packages.keys()
                            if name not in packages.keys()]
        
        print(missing_packages)

        assert not missing_packages, \
            f'Following packages are missing:\n{missing_packages}'

    def test_missmatched_versions(self, req_packages, packages):
        present_packages = {name: version for name, version in req_packages.items()
                            if name in packages.keys()}

        missmatched_versions = {name: version for name, version in present_packages.items()
                                if packages[name] != version }
        
        print(missmatched_versions)

        assert not missmatched_versions, \
            f'Found missmatched versions for the following Packages:\n{missmatched_versions.keys()}'