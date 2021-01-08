# python setup.py bdist_wheel
from setuptools import find_packages, setup

setup(
    name='Fasttorch',
    version='1.0.2',
    author="Legend",
    description="A keras-like library for pytorch.",
    packages=find_packages(),
    # 数据文件全部打包
    package_data={"":["*"]},
    # 自动包含受版本控制(svn/git)的数据文件
    include_package_data=True,
    zip_safe=False,
    # 设置依赖包
    install_requires=[
        'scikit-learn', 'torch', 'tqdm', 'pandas',
    ],
)

