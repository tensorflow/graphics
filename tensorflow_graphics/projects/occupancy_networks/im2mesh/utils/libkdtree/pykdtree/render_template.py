#!/usr/bin/env python

from mako.template import Template

mytemplate = Template(filename='_kdtree_core.c.mako')
with open('_kdtree_core.c', 'w') as fp:
    fp.write(mytemplate.render())
