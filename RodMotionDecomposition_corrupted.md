Rigid Motion Decomposition for Multiple Rods
Summary of Theory, Derivations, and Simulation Framework
1. Rigid Motions and Screw Motions

Any rigid motion in 
𝑅
3
R
3
 can be written as:

𝑥
′
=
𝑅
𝑥
+
𝑡
,
𝑅
∈
𝑆
𝑂
(
3
)
,
  
𝑡
∈
𝑅
3
.
x
′
=Rx+t,R∈SO(3),t∈R
3
.

By Chasles' theorem, every rigid motion is a screw motion:
rotation by 
𝜃
θ about some axis 
ℓ
ℓ, plus translation 
ℎ
h along that axis.

Given 
(
𝑅
,
𝑡
)
(R,t), the screw parameters are:

Axis direction

𝑛
n = unit eigenvector of 
𝑅
R with eigenvalue 1.

Rotation angle

cos
⁡
𝜃
=
1
2
(
t
r
(
𝑅
)
−
1
)
cosθ=
2
1
	​

(tr(R)−1).

Pitch (translation along axis)

ℎ
=
𝑛
⋅
𝑡
h=n⋅t.

Point on axis
Solve

(
𝐼
−
𝑅
)
𝑝
=
𝑡
−
(
𝑛
⋅
𝑡
)
𝑛
.
(I−R)p=t−(n⋅t)n.
2. Skew Lines & Invariance Under Screw Motion

Distance between skew lines 
𝐿
1
L
1
	​

 and 
𝐿
2
L
2
	​

:

𝑑
(
𝐿
1
,
𝐿
2
)
=
∣
(
𝑝
2
−
𝑝
1
)
⋅
(
𝑎
×
𝑏
)
∣
∥
𝑎
×
𝑏
∥
.
d(L
1
	​

,L
2
	​

)=
∥a×b∥
∣(p
2
	​

−p
1
	​

)⋅(a×b)∣
	​

.

Rigid motion is an isometry, so:

Cross products are preserved: 
𝑅
(
𝑎
×
𝑏
)
=
(
𝑅
𝑎
)
×
(
𝑅
𝑏
)
R(a×b)=(Ra)×(Rb)

Dot products are preserved

Translation cancels in differences

Thus:

𝑑
(
𝐿
1
′
,
𝐿
2
′
)
=
𝑑
(
𝐿
1
,
𝐿
2
)
.
d(L
1
′
	​

,L
2
′
	​

)=d(L
1
	​

,L
2
	​

).
3. When Does a Global Rigid Motion Exist?

Given many rods or points 
𝑟
𝑖
(
𝑠
,
𝑡
)
r
i
	​

(s,t), a global rigid motion exists iff:

∥
𝑟
𝑖
(
𝑠
1
,
𝑡
)
−
𝑟
𝑗
(
𝑠
2
,
𝑡
)
∥
is constant in time.
∥r
i
	​

(s
1
	​

,t)−r
j
	​

(s
2
	​

,t)∥is constant in time.

Equivalent instantaneous condition:

𝑑
𝑑
𝑡
∥
𝑟
𝑖
−
𝑟
𝑗
∥
2
=
0
∀
𝑖
,
𝑗
.
dt
d
	​

∥r
i
	​

−r
j
	​

∥
2
=0∀i,j.
Reconstruction method:

Fit best rigid motion using Kabsch algorithm

Test residuals = zero → perfect rigid motion

Residuals > 0 → internal deformations

4. Why Two Endpoints per Rod Is Not Enough Alone

A single rod gives 2 points 
𝑝
0
,
𝑝
1
p
0
	​

,p
1
	​

.
Mapping these two points does not determine a unique rigid motion:
we can freely rotate around the line 
𝑝
0
𝑝
1
p
0
	​

p
1
	​

.

However:

several rods → endpoints are not collinear

any three non-collinear points determine a unique rigid motion

Thus, with many rods:

Two endpoints per rod = plenty of constraints for a unique SE(3).

5. Twist Representation (
𝑠
𝑒
(
3
)
se(3))

Rigid body instantaneous velocity field:

𝑣
(
𝑥
)
=
𝜔
×
𝑥
+
𝑣
0
.
v(x)=ω×x+v
0
	​

.

The pair 
(
𝜔
,
𝑣
0
)
(ω,v
0
	​

) is a twist:

𝜉
=
(
𝜔


𝑣
0
)
∈
𝑅
6
.
ξ=(
ω
v
0
	​

	​

)∈R
6
.

Rigid transform from twist:

𝑇
=
exp
⁡
(
𝜉
^
)
,
𝜉
^
=
(
[
𝜔
]
×
	
𝑣
0


0
	
0
)
.
T=exp(
ξ
^
	​

),
ξ
^
	​

=(
[ω]
×
	​

0
	​

v
0
	​

0
	​

).
6. Best-Fit Global Rigid Motion of Many Rods

Given marker points 
𝑥
𝑖
(
𝑡
)
x
i
	​

(t) and velocities 
𝑣
𝑖
(
𝑡
)
v
i
	​

(t):

Fit the closest rigid motion:
𝑣
𝑖
rigid
=
𝜔
×
𝑥
𝑖
+
𝑣
0
v
i
rigid
	​

=ω×x
i
	​

+v
0
	​


by minimizing

∑
𝑖
𝑚
𝑖
∥
𝑣
𝑖
−
(
𝜔
×
𝑥
𝑖
+
𝑣
0
)
∥
2
.
i
∑
	​

m
i
	​

∥v
i
	​

−(ω×x
i
	​

+v
0
	​

)∥
2
.

This is a projection of the velocity field onto the 6D subspace of rigid motions.

Then:

𝑣
𝑖
def
=
𝑣
𝑖
−
𝑣
𝑖
rigid
.
v
i
def
	​

=v
i
	​

−v
i
rigid
	​

.

This gives a clean decomposition:

global motion = bulk drift + rotation

local motion = bending, shearing, entangling, rod rearrangement

7. Energy Decomposition

Total kinetic energy:

𝑇
=
1
2
∑
𝑖
𝑚
𝑖
∥
𝑣
𝑖
∥
2
.
T=
2
1
	​

i
∑
	​

m
i
	​

∥v
i
	​

∥
2
.

Insert 
𝑣
𝑖
=
𝑣
𝑖
rigid
+
𝑣
𝑖
def
v
i
	​

=v
i
rigid
	​

+v
i
def
	​

:

𝑇
=
𝑇
global
+
𝑇
def
,
T=T
global
	​

+T
def
	​

,

because the projection ensures:

∑
𝑖
𝑚
𝑖
 
𝑣
𝑖
rigid
⋅
𝑣
𝑖
def
=
0.
i
∑
	​

m
i
	​

v
i
rigid
	​

⋅v
i
def
	​

=0.

Meaning:

Global kinetic energy

𝑇
global
=
1
2
∑
𝑚
𝑖
∥
𝑣
𝑖
rigid
∥
2
.
T
global
	​

=
2
1
	​

∑m
i
	​

∥v
i
rigid
	​

∥
2
.

Deformational kinetic energy

𝑇
def
=
1
2
∑
𝑚
𝑖
∥
𝑣
𝑖
def
∥
2
.
T
def
	​

=
2
1
	​

∑m
i
	​

∥v
i
def
	​

∥
2
.

This is exactly the energy split used in:

continuum mechanics (rigid + strain energy),

multibody dynamics,

granular flows with “fluctuation” kinetic energy,

turbulence (Reynolds decomposition).

8. Python Pipeline Summary

Sample markers on rods (e.g. endpoints).

Compute velocities by finite difference.

Fit twist 
(
𝜔
,
𝑣
0
)
(ω,v
0
	​

) by least squares.

Compute

global velocity 
𝑣
𝑖
rigid
v
i
rigid
	​

,

deformational velocity 
𝑣
𝑖
def
v
i
def
	​

,

global + deformational kinetic energies.

Visualize rods and vectors.

(Optional) Fit full SE(3) using Kabsch to track positions across frames.

9. Visualization Insights

We showed that:

One rod alone → infinite rigid motions (free rotation about rod axis).

Two rods → endpoints give enough constraints → unique rigid motion.

The two-rod visualization makes this intuitively clear:

Rotate around rod A → rod B no longer aligns → only one SE(3) fits both rods simultaneously.

10. What We Can Package Next

I can generate a downloadable .zip file containing:

README.md (this summary)

rigid_motion_tutorial.py (all code: rods, twist fit, Kabsch, energy decomposition)

visualization scripts

examples + synthetic data generation

optional JAX version

optional screw-motion animation