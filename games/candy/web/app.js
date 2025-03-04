document.addEventListener('DOMContentLoaded', () => {
const grid = document.querySelector('.grid')
const scoreDisplay = document.getElementById('score')
const width = 8
const squares = []
let score = 0
let ini = true

const timerDisplay = document.getElementById('timer')
  let timeRemaining = 20    // 20-second timer
  let timerId   

const candyColors = [
    'url(images/red-candy.png)',
    'url(images/yellow-candy.png)',
    'url(images/orange-candy.png)',
    'url(images/purple-candy.png)',
    'url(images/green-candy.png)',
    'url(images/blue-candy.png)'
  ]

//create your board
function createBoard() {
  for (let i = 0; i < width*width; i++) {
    const square = document.createElement('div')
    square.setAttribute('draggable', true)
    square.setAttribute('id', i)
    let randomColor = Math.floor(Math.random() * candyColors.length)
    square.style.backgroundImage = candyColors[randomColor]
    grid.appendChild(square)
    squares.push(square)
  }
}


//drop candies once some have been cleared
// function moveIntoSquareBelow() {
//     for (i = 0; i < 56; i ++) {
//         if(squares[i + width].style.backgroundImage === '') {
//             squares[i + width].style.backgroundImage = squares[i].style.backgroundImage
//             squares[i].style.backgroundImage = ''
//             const firstRow = [0, 1, 2, 3, 4, 5, 6, 7]
//             const isFirstRow = firstRow.includes(i)
//             if (isFirstRow && (squares[i].style.backgroundImage === '')) {
//               let randomColor = Math.floor(Math.random() * candyColors.length)
//               squares[i].style.backgroundImage = candyColors[randomColor]
//             }
//         }
//     }
// }

function moveIntoSquareBelow() {
  for (let i = 0; i < width * (width - 1); i++) {
    if (squares[i + width].style.backgroundImage === '') {
      squares[i + width].style.backgroundImage = squares[i].style.backgroundImage;
      squares[i].style.backgroundImage = '';
    }
  }

    // Refill the first row properly
  for (let i = 0; i < width; i++) {
    if (squares[i].style.backgroundImage === '') {
      let randomColor = Math.floor(Math.random() * candyColors.length);
      squares[i].style.backgroundImage = candyColors[randomColor];
    }
  }
}


///Checking for Matches
//for row of Four
  function checkRowForFour() {
    for (i = 0; i < 60; i ++) {
      let rowOfFour = [i, i+1, i+2, i+3]
      let decidedColor = squares[i].style.backgroundImage
      const isBlank = squares[i].style.backgroundImage === ''

      const notValid = [5, 6, 7, 13, 14, 15, 21, 22, 23, 29, 30, 31, 37, 38, 39, 45, 46, 47, 53, 54, 55]
      if (notValid.includes(i)) continue

      if(rowOfFour.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)) {
        if (!ini) {
          score += 4
        }
        scoreDisplay.innerHTML = score
        rowOfFour.forEach(index => {
        squares[index].style.backgroundImage = ''
        })
      }
    }
  }

//for column of Four
  function checkColumnForFour() {
    for (i = 0; i < 39; i ++) {
      let columnOfFour = [i, i+width, i+width*2, i+width*3]
      let decidedColor = squares[i].style.backgroundImage
      const isBlank = squares[i].style.backgroundImage === ''

      if(columnOfFour.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)) {
        if (!ini) {
          score += 4
          
        }
        scoreDisplay.innerHTML = score
        columnOfFour.forEach(index => {
        squares[index].style.backgroundImage = ''
        })
      }
    }
  }

  //for row of Three
  function checkRowForThree() {
    for (i = 0; i < 61; i ++) {
      let rowOfThree = [i, i+1, i+2]
      let decidedColor = squares[i].style.backgroundImage
      const isBlank = squares[i].style.backgroundImage === ''

      const notValid = [6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55]
      if (notValid.includes(i)) continue

      if(rowOfThree.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)) {
        if (!ini) {
          score += 3

        }
        scoreDisplay.innerHTML = score
        rowOfThree.forEach(index => {
        squares[index].style.backgroundImage = ''
        })
      }
    }
  }

//for column of Three
  function checkColumnForThree() {
    for (i = 0; i < 47; i ++) {
      let columnOfThree = [i, i+width, i+width*2]
      let decidedColor = squares[i].style.backgroundImage
      const isBlank = squares[i].style.backgroundImage === ''

      if(columnOfThree.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)) {
        if (!ini){
          score += 3
        }
        scoreDisplay.innerHTML = score
        columnOfThree.forEach(index => {
        squares[index].style.backgroundImage = ''
        })
      }
    }
  }

  function settleInitialBoard() {
    let i = 0
    while (i < 10) {
      checkRowForFour()
      checkColumnForFour()
      checkRowForThree()
      checkColumnForThree()
      moveIntoSquareBelow()
      i++
    }
    ini = false
    moveIntoSquareBelow()
  }



// function startTimer() {
//   timerDisplay.innerHTML = `Time: ${timeRemaining}`

//   timerId = setInterval(() => {
//     timeRemaining--
//     timerDisplay.innerHTML = `Time: ${timeRemaining}`

//     if (timeRemaining <= 0) {
//       clearInterval(timerId)
//       endGame()
//     }
//   }, 1000)
// }


// function endGame() {
//   // Stop your match-check loop
//   clearInterval(matchCheckInterval);

//   // Disable dragging, etc. (Optional)
//   squares.forEach(square => {
//     square.removeEventListener('dragstart', dragStart)
//     square.removeEventListener('dragend', dragEnd)
//     square.removeEventListener('dragover', dragOver)
//     square.removeEventListener('dragenter', dragEnter)
//     square.removeEventListener('dragleave', dragLeave)
//     square.removeEventListener('drop', dragDrop)
//   })

//   // Show the pop-up with the final score (or any message)
//   alert(`Time's up! Final Score = ${score}`)
// }

  

createBoard()
scoreDisplay.innerHTML = score
settleInitialBoard()


// Dragging the Candy
let colorBeingDragged
let colorBeingReplaced
let squareIdBeingDragged
let squareIdBeingReplaced

squares.forEach(square => square.addEventListener('dragstart', dragStart))
squares.forEach(square => square.addEventListener('dragend', dragEnd))
squares.forEach(square => square.addEventListener('dragover', dragOver))
squares.forEach(square => square.addEventListener('dragenter', dragEnter))
squares.forEach(square => square.addEventListener('drageleave', dragLeave))
squares.forEach(square => square.addEventListener('drop', dragDrop))

function dragStart(){
    colorBeingDragged = this.style.backgroundImage
    squareIdBeingDragged = parseInt(this.id)
    // this.style.backgroundImage = ''
}

function dragOver(e) {
    e.preventDefault()
}

function dragEnter(e) {
    e.preventDefault()
}

function dragLeave() {
    this.style.backgroundImage = ''
}

function dragDrop() {
    colorBeingReplaced = this.style.backgroundImage
    squareIdBeingReplaced = parseInt(this.id)
    this.style.backgroundImage = colorBeingDragged
    squares[squareIdBeingDragged].style.backgroundImage = colorBeingReplaced
}



function isValidMatch() {
  // Check row of four
  for (let i = 0; i < 60; i++) {
    const rowOfFour = [i, i+1, i+2, i+3]
    const decidedColor = squares[i].style.backgroundImage
    const isBlank = (decidedColor === '')

    // Skip if near right edge
    const notValid = [5,6,7,13,14,15,21,22,23,29,30,31,37,38,39,45,46,47,53,54,55]
    if (notValid.includes(i)) continue

    if (
      rowOfFour.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)
    ) {
      return true
    }
  }

  // Check row of three
  for (let i = 0; i < 61; i++) {
    const rowOfThree = [i, i+1, i+2]
    const decidedColor = squares[i].style.backgroundImage
    const isBlank = (decidedColor === '')

    const notValid = [6,7,14,15,22,23,30,31,38,39,46,47,54,55]
    if (notValid.includes(i)) continue

    if (
      rowOfThree.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)
    ) {
      return true
    }
  }

  // Check column of four
  for (let i = 0; i < 39; i++) {
    const columnOfFour = [i, i+width, i+width*2, i+width*3]
    const decidedColor = squares[i].style.backgroundImage
    const isBlank = (decidedColor === '')

    if (
      columnOfFour.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)
    ) {
      return true
    }
  }

  // Check column of three
  for (let i = 0; i < 47; i++) {
    const columnOfThree = [i, i+width, i+width*2]
    const decidedColor = squares[i].style.backgroundImage
    const isBlank = (decidedColor === '')

    if (
      columnOfThree.every(index => squares[index].style.backgroundImage === decidedColor && !isBlank)
    ) {
      return true
    }
  }

  // If we get here, no matches of 3 or more
  return false
}

// Drag End
function dragEnd() {
  // Check if swapped candy is adjacent
  let validMoves = [
    squareIdBeingDragged - 1,
    squareIdBeingDragged - width,
    squareIdBeingDragged + 1,
    squareIdBeingDragged + width
  ]
  let validMove = validMoves.includes(squareIdBeingReplaced)

  if (squareIdBeingReplaced && validMove) {
    // After performing the swap, check for a match:
    if (!isValidMatch()) {
      // No match => revert swap
      squares[squareIdBeingReplaced].style.backgroundImage = colorBeingReplaced
      squares[squareIdBeingDragged].style.backgroundImage = colorBeingDragged
    }
    squareIdBeingReplaced = null
  } else if (squareIdBeingReplaced && !validMove) {
    // Not a valid move => revert swap
    squares[squareIdBeingReplaced].style.backgroundImage = colorBeingReplaced
    squares[squareIdBeingDragged].style.backgroundImage = colorBeingDragged
  } else {
    // No replacement => revert to original
    squares[squareIdBeingDragged].style.backgroundImage = colorBeingDragged
  }
}



// Checks carried out indefintely - Add Button to clear interval for best practise, or clear on game over/game won. If you have this indefinite check you can get rid of calling the check functions above.
window.setInterval(function(){
    checkRowForFour()
    checkColumnForFour()
    checkRowForThree()
    checkColumnForThree()
    moveIntoSquareBelow()
  }, 100)

startTimer()

})

