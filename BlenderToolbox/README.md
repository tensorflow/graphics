# Blender Toolbox

This is a set of Python scripts for rendering 3D shapes in [Blender 2.8](https://www.blender.org). These scripts are from my private codebase for rendering figures in my academic publications. To use them, make sure you have installed Blender 2.8 and you can run the demo by typing 
```
blender --background --python demo.py
```

This toolbox contains a set of stand alone demos in `./cycles/` to demonstrate different rendering effects. You can see the results of the demos in `./cycles/results/`. A step-by-step tutorial with full documentation is included in the `./cycles/tutorial.py`.

Before rendering a scene, you probably need to set up the default rendering devices in the user preferences (e.g., which GPU to use). You only need to set up the user preferences once, then the script should be able to detect the GPUs automatically in the future. To set up the rendering devices, open the blender, go to `Edit` > `Preferences` > `System`, then in the `Cycles Render Devices` select your preferred devices for rendering (e.g., select `CUDA` and check every GPUs on your computer). After setting up the devices, click the `Save Preference` on bottom left.

If any questions or recommendations, please contact me at hsuehtil@cs.toronto.edu. 
