from setuptools import setup

def readme():
    with open('README.md') as fp:
        return fp.read()

setup(name='pyspex',
      use_scm_version={
          "root": ".",
          "relative_to": __file__,
          "write_to": 'pyspex/version.py',
          "fallback_version": "0.1.0"},
      setup_requires=['setuptools_scm'],
      description='Software package to access SPEXone data products',
      long_description=readme(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License (standard 3-clause)',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Atmospheric Science',
      ],
      url='https://github.com/rmvanhees/pyspex.git',
      author='Richard van Hees',
      author_email='r.m.van.hees@sron.nl',
      maintainer='Richard van Hees',
      maintainer_email='r.m.van.hees@sron.nl',
      license='BSD-3-Clause',
      packages=[
          'pyspex',
          'pyspex.lib',
      ],
      scripts=[
          'scripts/spx_ccsds2l1a.py',
          'scripts/cre_orbit_l1a.py',
          'scripts/spx_csv2bin_tbl.py',
          'scripts/spx_dem2l1a.py',
          'scripts/spx_tif2l1a.py'
      ],
      install_requires=[
          'numpy>=1.17',
          'h5py>=2.10',
          'netCDF4>=1.5',
          'pys5p>=1.0'
      ],
      zip_safe=False)
