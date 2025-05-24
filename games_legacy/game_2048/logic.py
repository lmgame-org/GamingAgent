import random


def move(direction, board):
    """
    Call functions to move & merge in the specified direction.

    Parameters:
        direction (str): direction in which to move the tiles
        board (list): game board
    Returns:
        (list): updated board after move completion
    """
    if direction == "w":
        return moveUp(board)
    if direction == "s":
        return moveDown(board)
    if direction == "a":
        return moveLeft(board)
    if direction == "d":
        return moveRight(board)


def checkGameStatus(board, max_tile=2048):
    """
    Update the game status by checking if the max. tile has been obtained.

    Parameters:
        board (list): game board
        max_tile (int): tile number required to win, default = 2048
    Returns:
        (str): game status WIN/LOSE/PLAY
    """
    flat_board = [cell for row in board for cell in row]
    if max_tile in flat_board:
        # game has been won if max_tile value is found
        return "WIN"

    for i in range(4):
        for j in range(4):
            # check if a merge is possible
            if j != 3 and board[i][j] == board[i][j+1] or \
                    i != 3 and board[i][j] == board[i + 1][j]:
                return "PLAY"

    if 0 not in flat_board:
        return "LOSE"
    else:
        return "PLAY"


def fillTwoOrFour(board, iter=1):
    """
    Randomly fill 2 or 4 in available spaces on the board.

    Parameters:
        board (list): game board
        iter (int): number of times to repeat the process
    Returns:
        board (list): updated game board
    """
    for _ in range(iter):
        a = random.randint(0, 3)
        b = random.randint(0, 3)
        while(board[a][b] != 0):
            a = random.randint(0, 3)
            b = random.randint(0, 3)

        if sum([cell for row in board for cell in row]) in (0, 2):
            board[a][b] = 2
        else:
            board[a][b] = random.choice((2, 4))
    return board


def moveLeft(board):
    """
    Move and merge tiles to the left.

    Parameters:
        board (list): game board
    Returns:
        board (list): updated game board
    """
    # initial shift
    shiftLeft(board)

    # merge cells
    for i in range(4):
        for j in range(3):
            if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                board[i][j] *= 2
                board[i][j + 1] = 0
                j = 0

    # final shift
    shiftLeft(board)
    return board


def moveUp(board):
    """
    Move ane merge tiles upwards.

    Parameters:
        board (list): game board
    Returns:
        board (list): updated game board
    """
    board = rotateLeft(board)
    board = moveLeft(board)
    board = rotateRight(board)
    return board


def moveRight(board):
    """
    Move and merge tiles to the right.

    Parameters:
        board (list): game board
    Returns:
        board (list): updated game board
    """
    # initial shift
    shiftRight(board)

    # merge cells
    for i in range(4):
        for j in range(3, 0, -1):
            if board[i][j] == board[i][j - 1] and board[i][j] != 0:
                board[i][j] *= 2
                board[i][j - 1] = 0
                j = 0

    # final shift
    shiftRight(board)
    return board


def moveDown(board):
    """
    Move and merge tiles downwards.

    Parameters:
        board (list): game board
    Returns:
        board (list): updated game board
    """
    board = rotateLeft(board)
    board = moveLeft(board)
    shiftRight(board)
    board = rotateRight(board)
    return board


def shiftLeft(board):
    """
    Perform tile shift to the left.

    Parameters:
        board (list): game board
    """
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = nums
        board[i].extend([0] * (4 - count))


def shiftRight(board):
    """
    Perform tile shift to the right.

    Parameters:
        board (list): game board
    """
    # remove 0's in between numbers
    for i in range(4):
        nums, count = [], 0
        for j in range(4):
            if board[i][j] != 0:
                nums.append(board[i][j])
                count += 1
        board[i] = [0] * (4 - count)
        board[i].extend(nums)


def rotateLeft(board):
    """
    90 degree counter-clockwise rotation.

    Parameters:
        board (list): game board
    Returns:
        b (list): new game board after rotation
    """
    b = [[board[j][i] for j in range(4)] for i in range(3, -1, -1)]
    return b


def rotateRight(board):
    """
    270 degree counter-clockwise rotation.

    Parameters:
        board (list): game board
    Returns:
        (list): new game board after rotation
    """
    b = rotateLeft(board)
    b = rotateLeft(b)
    return rotateLeft(b)
