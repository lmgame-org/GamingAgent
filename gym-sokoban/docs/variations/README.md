# Variations

| Fixed Targets | Two Player | Push&Pull |
| :---: | :---: | :---: 
| ![Game 1](/docs/Animations/fixedTargets_solved_0.gif?raw=true) | ![Game 2](/docs/Animations/TwoPlayer_solved_2.gif?raw=true) | ![Game 3](/docs/Animations/pushAndPull_solved_0.gif?raw=true) |


Besides the regular game of Sokoban, this repository implements or will implement variations, which might make the game easier or more complicated. Except noted differently the variations do not implement a Tiny-World version.

## Variations
| Variation | Summary | Expected Difficulty | Example | Tiny World | Status | Details |
| ---       | :---:   | :---:               | :---:   | :---: | :---: | :---: |
| Fixed Targets | Every box has to be pushed on the target with the same color. | More difficult | ![Fixed-Targets](/docs/rooms/Sokoban-Fixed-Targets-Example.png) | Yes | implemented | [ReadMe](/docs/variations/FixedTargets.md) |
| Multiple Player | There are two players in the room. Every round one of the two players can be used. There is no order of moves between the two players. | More difficult | ![TwoPlayer](/docs/rooms/TwoPlayer-Sokoban-v2.png) | Yes | planned | [ReadMe](/docs/variations/TwoPlayer.md) |
| Push&Pull | The player can not only push the boxes, but also pull them. Therefor no more irreversible moves exist. | Easier | ![PushAndPull-Targets](/docs/rooms/Sokoban-v1.png) | Yes | implemented | [ReadMe](/docs/variations/FixedTargets.md) |

## Tiny World

Currently the tiny world variations are only available for the regular Sokoban game.
