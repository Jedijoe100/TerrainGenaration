The core concepts are there for a biome map, however, the values need to be refined to get realistic planetoids.
[/image/data_figure_1.png]
This is the current best run.

## Method
The aim is to generate a biome map for a planet.
To do this there are a number of layers that are used: Altitude, Temperature, Humidity, Precipitation and Water Level.
### Altitude Map
The Altitude map is the most involved part.
It starts by generating a set of points corresponding to plates on a 3d grid before mapping plates to the points on the sphere.
Each plate is given a velocity and a plate type and is moved periodically during the simulation.
Between plate movement there is a very simple weather model to provide precipitation and allow for erosion over the plates.

### Temperature and Humidity
The temperature map is modified by two main methods, solar energy and radiation.
Solar energy is generated based on the time of year (changes as the simulation progresses), latitude and whether or not there are clouds.
Clouds are calculated on how much humidity there is based on the temperature.
Humidity is simply a function of temperature, water level and whether the altitude is below sea level.
Temperature and Humidity has two layers at each point to allow for weather cells in the weather simulation.

### Weather, Precipitation and Water Level
Wind is a very simple model that uses the coriolis effect and weather cells.
It is generated at the start and there is no modification to the wind patterns as the simulation progresses.
However, a blurring effect is applied to the temperature, humidity and water level to get a smoother gradient between these functions.
Precipitation is generated based on whether or not the humidity exceeds the temperature capacity.
The precipitation is then added to the water level which then flows down to the lowest point.
The water can collect in local minimum forming lakes.

### Biome classification
Biomes are classified from a csv file that dictates their colour and what characteristic that that biome has.
These characteristics include: what proportion of a year are spent at diffent temperature, what proportion of a year are spent with diffent temperatures, what altitude classification does the region have and what is the water content.
Each point is then plotted on this multi dimensional grid and biome is selected depending on which biome is closest to that point.

## Technical Features
The whole system is developed in python and can take 200+ seconds on a modest computer.
To allow for a more comprehensive understanding of what the terrain looks like there is a feature which allows for worlds to be exported to a minecraft world using amulet (incomplete).
Biomes are defined using a csv to allow for easy customisation.

## Future Ideas/Plans
- Rivers and other micro features (like volcanos)
- More realistic weather (either using dust as cloud seeds, using a fluid simulation for the weather or ocean currents)
- Terrain layers (like sand, gravel and stone)
- Different planet shapes (squashed, donut, tidally locked, ring world, etc...)
