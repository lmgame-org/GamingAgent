# Variation: Push&Pull

| Example Game 1 | Example Game 2 | Example Game 3 |
| :---: | :---: | :---: 
| ![Game 1](/docs/Animations/pushAndPull_solved_0.gif?raw=true) | ![Game 2](/docs/Animations/pushAndPull_solved_1.gif?raw=true) | ![Game 3](/docs/Animations/pushAndPull_solved_2.gif?raw=true) |


## 1. Idea
This variation follows the same rules as the regular game with on exception: 
The player can not only push boxes, but also pull the boxes. 
Because of that, the number of available actions increase to 13. The action mapping is the following.

 
 | Action       | ID    | 
 | --------     | :---: | 
 | No Operation |  0    | 
 | Push Up      |  1    |  
 | Push Down    |  2    | 
 | Push Left    |  3    |   
 | Push Right   |  4    |   
 | Move Up      |  5    |
 | Move Down    |  6    |
 | Move Left    |  7    |
 | Move Right   |  8    |
 | Pull Up      |  9    |
 | Pull Down    |  10   |
 | Pull Left    |  11   |
 | Pull Right   |  12   |

The conditions for winning do not change. Also the room generation algorithm is the same as before.

## 2. Room Configurations
Similar to the regular game there exist multiple room configurations, but TinyWorld configurations.

| Room Id | Grid-Size | Pixels | #Boxes | Example | 
| ---     | :---:      | :---: | :---:   | :---: | 
| PushAndPull-Sokoban-v0 | 10x10 | 160x160 | 3 | ![PushAndPull-Sokoban-v0](/docs/rooms/Sokoban-v0.png)  | 
| PushAndPull-Sokoban-v1 | 10x10 | 160x160 | 4 | ![PushAndPull-Sokoban-v1](/docs/rooms/Sokoban-v1.png)   | 
| PushAndPull-Sokoban-v2 | 7x7 | 112x112 | 2 | ![PushAndPull-Sokoban-v2](/docs/rooms/Sokoban-small-v0.png)  |
| PushAndPull-Sokoban-v3 | 7x7 | 112x112 | 3 | ![PushAndPull-Sokoban-v3](/docs/rooms/Sokoban-small-v1.png)  |
| PushAndPull-Sokoban-v4 | 13x11 | 208x176 | 4 | ![PushAndPull-Sokoban-v4](/docs/rooms/Sokoban-large-v0.png)  |
| PushAndPull-Sokoban-v5 | 13x11 | 208x176 | 5 | ![PushAndPull-Sokoban-v5](/docs/rooms/Sokoban-large-v1.png)  |


