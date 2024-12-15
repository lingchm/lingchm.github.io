---
layout: page
title: Gymbal Lock and Quaternions  
description: 
img:
importance: 4
category: molecular dynamics
---

# Gymbal Lock

Gimbal lock is a phenomenon that occurs when representing 3D orientation or rotation using Euler angles, which are a set of three angles that describe the orientation of a rigid body relative to a fixed coordinate system. The phenomenon happens when two of the three rotational axes become aligned, resulting in a loss of one degree of freedom in the system.

To understand gimbal lock, let's break it down:

Euler Angles and Rotations:
Euler angles describe 3D rotation in terms of three sequential rotations around specific axes (usually called pitch, yaw, and roll):

Pitch: Rotation around the X-axis.
Yaw: Rotation around the Y-axis.
Roll: Rotation around the Z-axis.
These rotations are applied sequentially to an object to change its orientation in 3D space.

The Gimbal Mechanism:
To visualize rotations in 3D, a gimbal system can be used. A gimbal consists of three nested rings (each representing one axis of rotation), with each ring rotating freely around one of the three axes. The outer gimbal controls yaw, the middle one controls pitch, and the innermost one controls roll.

In a properly functioning gimbal system, you can rotate the object in any direction by adjusting the three gimbals independently. However, gimbal lock occurs when two of these gimbals become aligned and rotate together. This alignment leads to a situation where one degree of freedom is lost, and the system can no longer represent certain rotations.

Why Does Gimbal Lock Happen?
Gimbal lock happens when the pitch angle becomes exactly ±90°. At this point:

The yaw and roll axes become aligned, meaning you can't rotate independently around those two axes anymore.
The system loses one degree of freedom because you can no longer distinguish between certain rotations.
In other words, when the pitch angle is ±90°, rotating the object around the yaw or roll axis has the same effect, meaning you lose the ability to rotate in a fully independent 3D manner.

Example:
Imagine an aircraft or a camera, for example, that uses Euler angles to specify its orientation. If the pitch angle becomes ±90°, the aircraft or camera may "lose" its ability to rotate independently around certain axes (yaw and roll). This means that any further rotation will not result in a unique orientation, as the system can't distinguish between them.

Why Quaternions Help:
Quaternions are often used to represent 3D rotations as they do not suffer from gimbal lock. A quaternion is a four-dimensional extension of complex numbers, and unlike Euler angles, they allow for smooth, continuous rotations in 3D without the risk of losing a degree of freedom. Using quaternions, you can represent any rotation in 3D space without the issues associated with gimbal lock.

In Summary:
Gimbal lock occurs when two axes of rotation become aligned in a 3D system using Euler angles, causing the system to lose one degree of freedom and making certain rotations indistinguishable.
It happens when the pitch angle is exactly ±90°, which causes the yaw and roll axes to align.
This is why quaternions are often preferred for 3D orientation in applications like computer graphics, robotics, and aerospace, as they avoid gimbal lock and provide smooth, continuous rotations.