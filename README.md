# Integración de Geodésicas Nulas en la Métrica de Schwarzschild

Este es un pequeño y humilde proyecto que realicé en mi estancia en el pregrado en física. Consiste en una simulación y visualización del comportamiento de las geodésicas nulas alrededor de un agujero negro de Schwarzschild.

Combina un cálculo simbólico con SymPy para obtener los símbolos de Christoffel a partir del tensor métrico y un entorno de cálculo numérico acelerado por GPU mediante CuPy para integrar las ecuaciones geodésicas utilizando el método de  Runge-Kutta 4. Finalizando con una prueba de la conservación de la norma nula y la generación de una animación 3D.

Notar que la conservación de la norma nula deja de verificarse numéricamente cuando el rayo se aproxima al horizonte de eventos. Este resultado es de carácter numérico, ya que teóricamente la norma nula se conserva a lo largo de toda la geodésica, incluyendo el horizonte de eventos. El comportamiento observado está relacionado con la singularidad de las coordenadas de Schwarzschild en el horizonte, que no corresponde  auna singularidad física del espacio-tiempo. Teóricamente, se puede solventar esta dificultad al trabajar con las coordenadas de Kruskal-Szekeres. Sin embargo, en el momento que realicé esta simulacion, por simpleza opté conservar la formulación original y detener la integración cerca del horizonte.

Sería interesante adaptar la simulación para las coordenadas de Kruskal-Szekeres y comparar los resultados con la implementación actual.

---

## Características

- Cálculo Simbólico Automático: Deducción de los símbolos de Christoffel a partir de la matriz métrica mediante SymPy.
- Aceleración por GPU: Integración numérica del sistema de ecuaciones diferenciales usando CuPy para el procesamiento con CUDA.
- Integrador Numérico: Implementación de un integrador Runge-Kutta de 4to orden (RK4) adaptado a las condiciones de frontera (frenado   cerca del horizonte de eventos r = 2M).
- Verificación Física: Gráfica de control que verifica la geodésica nula a lo largo de los pasos de integración.
- Animación 3D: Renderizado interactivo y exportación a formato GIF de los haces de luz.

---

## Requisitos

Tarjeta gráfica NVIDIA con soporte para CUDA.

- Python 3.8+
- Driver de NVIDIA actualizado

---

# Integration of Null Geodesics in the Schwarzschild Metric

This is a small and humble project I developed during my undergraduate studies in physics. It consists of a simulation and visualization of the behavior of null geodesics around a Schwarzschild black hole.

It combines symbolic computation with SymPy to obtain the Christoffel symbols from the metric tensor, along with a GPU-accelerated numerical computing environment using CuPy to integrate the geodesic equations via the fourth-order Runge-Kutta method (RK4). Finally, it includes a test of null norm conservation and the generation of a 3D animation.

Note that null norm conservation ceases to hold numerically as the ray approaches the event horizon. This result is a numerical effect, since theoretically the null norm is conserved along the entire geodesic, including at the event horizon. The observed behavior is related to the singularity of Schwarzschild coordinates at the horizon, which does not correspond to a physical singularity of spacetime. Theoretically, this difficulty can be overcome by working with Kruskal-Szekeres coordinates. However, when I developed this simulation, I chose to retain the original formulation for simplicity and stop the integration near the horizon.

It would be interesting to adapt the simulation to Kruskal-Szekeres coordinates and compare the results with the current implementation.

---

## Features

- Automatic Symbolic Computation: Automatic derivation of the Christoffel symbols from the metric tensor using SymPy.
- GPU Acceleration: Numerical integration of the system of differential equations using CuPy for CUDA processing.
- Numerical Integrator: Implementation of a fourth-order Runge-Kutta integrator (RK4) adapted to boundary conditions (stopping near the event horizon at r = 2M).
- Physical Verification: A diagnostic plot monitoring null geodesic throughout the integration steps.
- 3D Animation: Interactive rendering and GIF export of light-ray trajectories.

---

## Requirements

An NVIDIA graphics card with CUDA support.

- Python 3.8+
- Up-to-date NVIDIA driver

---
