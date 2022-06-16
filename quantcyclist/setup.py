PACKAGE_NAME = 'quantcyclist'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(PACKAGE_NAME, parent_package, top_path)
    
    # pure python packages
    config.add_subpackage('compute')
    config.add_subpackage('data')
    config.add_subpackage('io')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())