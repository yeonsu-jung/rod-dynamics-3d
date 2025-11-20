# Rigid Body Velocity–Verlet Integration (Correct Form)

## Overview
This document summarizes a correct Velocity–Verlet (kick–drift–kick) integrator for rigid bodies, including translation and rotation, proper frame handling, and gyroscopic terms.

---

## 1. Translational Dynamics

Given mass \(m\), position **x**, velocity **v**, force **f**, and gravity **g**:

\[
a = \frac{f}{m} + g.
\]

Velocity–Verlet steps:

1. **Half-step velocity**  
   \[
   v_{n+1/2} = v_n + \frac{h}{2} a_n
   \]

2. **Full-step position**  
   \[
   x_{n+1} = x_n + h v_{n+1/2}
   \]

3. **Recompute forces at new state**  

4. **Second half-step velocity**  
   \[
   v_{n+1} = v_{n+1/2} + \frac{h}{2} a_{n+1}
   \]

---

## 2. Rotational Dynamics (Correct)

A rigid body with quaternion **q**, angular velocity **ω**, and body-frame inertia tensor \(I_\text{body}\):

### 2.1 Convert inertia into world frame
\[
I_\text{world} = R I_\text{body} R^\top,\qquad 
I_\text{world}^{-1} = R I_\text{body}^{-1} R^\top
\]
where \(R\) is the rotation matrix of quaternion **q**.

### 2.2 Angular dynamics with gyroscopic term
Euler’s rigid-body equation:
\[
I \dot{\omega} + \omega \times (I\omega) = \tau
\]

Solve for angular acceleration:
\[
\dot{\omega}
= I_\text{world}^{-1} \left(\tau - \omega \times (I_\text{world} \omega)\right)
\]

### 2.3 Velocity–Verlet form for rotation

1. **Half-step angular velocity:**
\[
\omega_{n+1/2}
= \omega_n + \frac{h}{2} \dot{\omega}_n
\]

2. **Orientation drift using quaternion exponential:**

Let \(w = \|\omega_{n+1/2}\|\), axis = \(\omega_{n+1/2}/w\)

\[
\theta = w h
\]

Quaternion update:
\[
q_{n+1} = \exp\left(\frac{\theta}{2} \, \text{axis}\right) q_n
\]

3. **Recompute torque at new orientation**

4. **Second half-step angular velocity:**
\[
\omega_{n+1}
= \omega_{n+1/2} + \frac{h}{2} \dot{\omega}_{n+1}
\]

---

## 3. Notes

- If inertia is spherical (scalar \(I\)), cross-term vanishes and rotation simplifies.
- Damping may be applied *after* the second half-step.
- Contact forces and torques must be recomputed between the two halves.
- Using half-step angular velocity for quaternion integration greatly improves stability.

---

## 4. Minimal Pseudocode

```
kick_half():
    v += (f/m + g) * (0.5*h)
    w += wdot(q, w, tau) * (0.5*h)

drift():
    x += v * h
    q = normalize( quatExp(w * h) * q )

recompute_forces_and_torques()

kick_second_half():
    v += (f/m + g) * (0.5*h)
    w += wdot(q, w, tau) * (0.5*h)
```

Where:

```
wdot(q, w, tau):
    R = rotationMatrix(q)
    Iw = I_world * w
    return I_world_inv * (tau - cross(w, Iw))
```
