#!/usr/bin/env python3

import pathlib

import click

from ballet.util.code import blacken_code
from ballet.util.fs import spliceext

def line_defines_feature(line):
    return 'Feature(' in line


def line_appends_feature(line):
    return '.append(' in line


def split_features_in_notebook(notebookpath):
    p = pathlib.Path(notebookpath)
    with p.open('r') as f:
        content = f.readlines()

    feature_content = []
    for line in content:
        if line_defines_feature(line):
            feature_content.append(line)
            yield '\n'.join(feature_content)
            feature_content = []
        elif line_appends_feature(line):
            continue
        else:
            feature_content.append(line)


def get_feature_imports():
    return 'from ballet import Feature\nimport ballet.eng'


def write_feature_content(content, handle):
    content = get_feature_imports() + content
    content = blacken_code(content)
    handle.write(content)


@click.command()
@click.option('--notebooks', type=str)
def main(notebooks):
    if notebooks is not None:
        notebooks = list(map(int, notebooks.split(',')))
    else:
        notebooks = range(1, 9+1)
    for notebook in notebooks:
        notebookpath = pathlib.Path('notebooks_features',
                'notebook_{:02d}.py'.format(notebook))
        for i, feature_content in enumerate(
                split_features_in_notebook(notebookpath)):
            print(f'Processing user {notebook}, feature {i}...')
            dst = pathlib.Path('features',
                               'user_{:02d}'.format(notebook),
                               'feature_{:02d}.py'.format(i+1))
            if dst.exists():
                dst = pathlib.Path(spliceext(dst, '_new'))

            print(f'Writing content to {dst}')
            with dst.open('w') as f:
                write_feature_content(feature_content, f)


if __name__ == '__main__':
    main()
