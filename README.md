The core concepts are there for a biome map, however, the values need to be refined to get realistic planetoids.

## Results
![biomes_seed_1](https://github.com/Jedijoe100/TerrainGenaration/assets/30392586/5462a235-6540-42e2-8458-16f565feab48)
![data_figure_1](https://github.com/Jedijoe100/TerrainGenaration/assets/30392586/966cbec2-84a8-4ae9-9a44-6087eb2a3981)

## Requirements
- Numpy
- Scipy
- pandas
- matplotlib
- pillow
- amulet-core

## Method
The aim is to generate a biome map for a planet.
To do this there are a number of layers that are used: Altitude, Temperature, Humidity, Precipitation and Water Level.
### Altitude Map
The Altitude map is the most involved part.
It starts by generating a set of points corresponding to plates on a 3d grid before mapping plates to the points on the sphere.
Each plate is given a velocity and a plate type and is moved periodically during the simulation.
Between plate movement, there is a straightforward weather model to provide precipitation and allow for erosion over the plates.

### Temperature and Humidity
The temperature map is modified by two main methods, solar energy and radiation.
Solar energy is generated based on the time of year (changes as the simulation progresses), latitude and whether or not there are clouds.
Clouds are calculated on how much humidity there is based on the temperature.
Humidity is simply a function of temperature, water level and whether the altitude is below sea level.
Temperature and Humidity have two layers at each point to allow for weather cells in the weather simulation.

### Weather, Precipitation and Water Level
Wind is a very simple model that uses the Coriolis effect and weather cells.
It is generated at the start and there is no modification to the wind patterns as the simulation progresses.
However, a blurring effect is applied to the temperature, humidity and water level to get a smoother gradient between these functions.
Precipitation is generated based on whether or not the humidity exceeds the temperature capacity.
The precipitation is then added to the water level which then flows down to the lowest point.
The water can be collected in local minimum-forming lakes.

### Biome classification
Biomes are classified from a csv file that dictates their colour and what characteristics that biome has.
These characteristics include: what proportion of a year is spent at different temperatures, what proportion of a year is spent with different temperatures, what altitude classification the region has and what is the water content.
Each point is then plotted on this multi-dimensional grid and a biome is selected depending on which biome is closest to that point.

## Technical Features
The whole system is developed in python and can take 200+ seconds on a modest computer.
To allow for a more comprehensive understanding of what the terrain looks like there is a feature that allows for worlds to be exported to a Minecraft world using amulet.
Biomes are defined using a CSV to allow for easy customisation.

## Future Ideas/Plans
- Rivers and other micro features (like volcanos)
- More realistic weather (either using dust as cloud seeds, using a fluid simulation for the weather or ocean currents)
- Terrain layers (like sand, gravel and stone)
- Different planet shapes (squashed, doughnut (toroidal), tidally locked, ringworld, etc...)
