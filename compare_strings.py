bash_python_path = ['', '/opt/paraviewopenfoam410/lib/paraview-4.1', '/opt/paraviewopenfoam410/lib/paraview-4.1/site-packages', '/usr/lib/python2.7/dist-packages/vtk', '/opt/paraviewopenfoam410/lib/paraview-4.1/site-packages/vtk', '/usr/lib/python27.zip', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-linux2', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/pymodules/python2.7', '/usr/lib/python2.7/dist-packages/ubuntu-sso-client']

ipython_path = ['', '/usr/bin', '/opt/paraviewopenfoam410/lib/paraview-4.1', '/opt/paraviewopenfoam410/lib/paraview-4.1/site-packages', '/usr/lib/python2.7/dist-packages/vtk', '/opt/paraviewopenfoam410/lib/paraview-4.1/site-packages/vtk', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/pymodules/python2.7', '/usr/lib/python2.7/dist-packages/ubuntu-sso-client', '/usr/lib/python2.7/dist-packages/IPython/extensions']

new_list = list(set(ipython_path) - set(bash_python_path))
print new_list
    

