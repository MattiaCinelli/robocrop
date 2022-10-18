# robocrop

A toy project to practice the creation of gym-like environments. 

In this project a robot has to plant and harvest crops in different settings and challenges.

Clone and install the package
```bash
pip install -e .
```


## Version 1
<img src="commons\RobocropV1.svg" width=300 height=256 align='right'>
To succeed the algorithm has to 
Plow -> Seed -> Water -> Harvest

Try this code to test it:


## Version 1.1
<img src="commons\RobocropV1.1.svg" width=550 height=256 align='right'>
To succeed the algorithm has to 
Plow -> Seed -> Water -> Water -> Harvest


## Version 2
Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step

Observation:
Between 0 and 3: Empty, Seeded, Small plant, Ready for harvest
Soil moisture: Between 0 - 100. 100 for each watering, decrease 25 point every step -->