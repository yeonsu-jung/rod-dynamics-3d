# Major update (July 2026)

The goal is to make it more specific to the publication, `2606.03952v1.pdf', by cleaning up obsolete features that I've tried. This paper focuses on rods (spherocylinders), NSC (currently, but I want to benchmark contact models effortlessly) in a free boundary.

## List of issues and possible improvements

* We will only consider spherocylinders. So perhaps, it's time to purge all implementations to compute distances between rigid bodies. (But, will having those distance computation create a lot of issues? or we can simply keep them?)
* Too many cli options (but also we need to keep necessary ones)
* Effortless change of contact models is essential. We want to choose one-sided harmonic, NSC, mujoco-like (not yet implemented; we may just use mujoco to do that with a reasonable number of rods around 200), hertz-mindlin (quite optional).
* I didn't go through a thorough benchmark between cpu and gpu. How faithful they are, gpu gains, further optimization, etc.
* There is implementation for periodic boundary condition. Even though we didn't use PBC here, I want to keep it if things are not complicated.
