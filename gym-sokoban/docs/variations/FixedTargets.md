# Variation: Fixed Targets

| Example Game 1 | Example Game 2 | Example Game 3 |
| :---: | :---: | :---: 
| ![Game 1](/docs/Animations/fixedTargets_solved_0.gif?raw=true) | ![Game 2](/docs/Animations/fixedTargets_solved_4.gif?raw=true) | ![Game 3](/docs/Animations/fixedTargets_solved_5.gif?raw=true) |


## 1. Idea
In this variation the player not only has to push all boxes on the targets, but also push every box on specific target.
The boxes and targets are color coded, such that for example the blue box needs to end up on the blue target. 

## 2. Rules
The rules in general stay the same, which means the player has to push all boxes on the targets. There are two changes.
First a player only gets a reward if a box ends up on its specific target. 
Second the game end when every box are placed on their specific targets.

## 3. Room Configurations
Similar to the regular Sokoban there exist multiple room configurations.

| Room Id | Grid-Size | Pixels | #Boxes | Example | 
| ---     | :---:      | :---: | :---:   | :---: | 
| FixedTarget-Sokoban-v0 | 10x10 | 160x160 | 3 | ![FixedTarget-Sokoban-v0](/docs/rooms/FixedTarget-Sokoban-v0.png)  | 
| FixedTarget-Sokoban-v1 | 10x10 | 160x160 | 4 | ![FixedTarget-Sokoban-v1](/docs/rooms/FixedTarget-Sokoban-v1.png)   | 
| FixedTarget-Sokoban-v2 | 7x7 | 112x112 | 2 | ![FixedTarget-Sokoban-v2](/docs/rooms/FixedTarget-Sokoban-v2.png)  |
| FixedTarget-Sokoban-v3 | 7x7 | 112x112 | 3 | ![FixedTarget-Sokoban-v3](/docs/rooms/FixedTarget-Sokoban-v3.png)  |

