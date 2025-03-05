#!../bin/python

import sys
import pygame
import string
import queue
import json
import os
import copy

CACHE_DIR = "cache/boxxel"

original_size = 32  # Original tile size
scale_factor = 1  # Default scaling factor

wall_original = pygame.image.load('games/boxxel/images/wall.png')
floor_original = pygame.image.load('games/boxxel/images/floor.png')
box_original = pygame.image.load('games/boxxel/images/box.png')
box_docked_original = pygame.image.load('games/boxxel/images/box_docked.png')
worker_original = pygame.image.load('games/boxxel/images/worker.png')
worker_docked_original = pygame.image.load('games/boxxel/images/worker_dock.png')
docker_original = pygame.image.load('games/boxxel/images/dock.png')

_last_saved_matrix = None

def scale_images():
    global wall, floor, box, box_docked, worker, worker_docked, docker
    new_size = int(original_size * scale_factor)

    wall = pygame.transform.scale(wall_original, (new_size, new_size))
    floor = pygame.transform.scale(floor_original, (new_size, new_size))
    box = pygame.transform.scale(box_original, (new_size, new_size))
    box_docked = pygame.transform.scale(box_docked_original, (new_size, new_size))
    worker = pygame.transform.scale(worker_original, (new_size, new_size))
    worker_docked = pygame.transform.scale(worker_docked_original, (new_size, new_size))
    docker = pygame.transform.scale(docker_original, (new_size, new_size))

def save_matrix(matrix, filename='game_state.json'):
    global _last_saved_matrix
    filename = os.path.join(CACHE_DIR, filename)
    if matrix == _last_saved_matrix:
        return  # No change, so do nothing
    _last_saved_matrix = copy.deepcopy(matrix)
    temp_filename = filename + '.tmp'
    with open(temp_filename, 'w') as f:
        json.dump(matrix, f)
    os.replace(temp_filename, filename)
    print("Matrix saved to JSON.")


class game:
    def is_valid_value(self, char):
        if ( char == ' ' or  # floor
             char == '#' or  # wall
             char == '@' or  # worker on floor
             char == '?' or  # dock
             char == '*' or  # box on dock
             char == '$' or  # box
             char == '+' ):  # worker on dock
            return True
        else:
            return False

    def __init__(self, filename, level):
        self.queue = queue.LifoQueue()
        self.matrix = []
        if level < 1 or level > 52:
            print("ERROR: Level " + str(level) + " is out of range")
            sys.exit(1)
        else:
            with open(filename, 'r') as file:
                level_found = False
                for line in file:
                    if not level_found:
                        if "Level " + str(level) == line.strip():
                            level_found = True
                    else:
                        if line.strip() != "":
                            row = []
                            for c in line:
                                if c != '\n' and self.is_valid_value(c):
                                    row.append(c)
                                elif c == '\n':
                                    continue
                                else:
                                    print("ERROR: Level " + str(level) + " has invalid value " + c)
                                    sys.exit(1)
                            self.matrix.append(row)
                        else:
                            break

    def load_size(self):
        x = 0
        y = len(self.matrix)
        for row in self.matrix:
            if len(row) > x:
                x = len(row)
        return (x * 32, y * 32)

    def get_matrix(self):
        return self.matrix

    def print_matrix(self):
        for row in self.matrix:
            for char in row:
                sys.stdout.write(char)
                sys.stdout.flush()
            sys.stdout.write('\n')

    def get_content(self, x, y):
        return self.matrix[y][x]

    def set_content(self, x, y, content):
        if self.is_valid_value(content):
            self.matrix[y][x] = content
        else:
            print("ERROR: Value '" + content + "' to be added is not valid")

    def worker(self):
        x = 0
        y = 0
        for row in self.matrix:
            for pos in row:
                if pos == '@' or pos == '+':
                    return (x, y, pos)
                else:
                    x += 1
            y += 1
            x = 0

    
    def can_move(self,x,y):
        return self.get_content(self.worker()[0]+x,self.worker()[1]+y) not in ['#','*','$']

    def next(self,x,y):
        return self.get_content(self.worker()[0]+x,self.worker()[1]+y)

    def can_push(self,x,y):
        return (self.next(x,y) in ['*','$'] and self.next(x+x,y+y) in [' ','?'])

    def is_completed(self):
        for row in self.matrix:
            for cell in row:
                if cell == '$':
                    return False
        return True

    def move_box(self,x,y,a,b):
#        (x,y) -> move to do
#        (a,b) -> box to move
        current_box = self.get_content(x,y)
        future_box = self.get_content(x+a,y+b)
        if current_box == '$' and future_box == ' ':
            self.set_content(x+a,y+b,'$')
            self.set_content(x,y,' ')
        elif current_box == '$' and future_box == '?':
            self.set_content(x+a,y+b,'*')
            self.set_content(x,y,' ')
        elif current_box == '*' and future_box == ' ':
            self.set_content(x+a,y+b,'$')
            self.set_content(x,y,'?')
        elif current_box == '*' and future_box == '?':
            self.set_content(x+a,y+b,'*')
            self.set_content(x,y,'?')

    def unmove(self):
        if not self.queue.empty():
            movement = self.queue.get()
            if movement[2]:
                current = self.worker()
                self.move(movement[0] * -1,movement[1] * -1, False)
                self.move_box(current[0]+movement[0],current[1]+movement[1],movement[0] * -1,movement[1] * -1)
            else:
                self.move(movement[0] * -1,movement[1] * -1, False)

    def move(self,x,y,save):
        if self.can_move(x,y):
            current = self.worker()
            future = self.next(x,y)
            if current[2] == '@' and future == ' ':
                self.set_content(current[0]+x,current[1]+y,'@')
                self.set_content(current[0],current[1],' ')
                if save: self.queue.put((x,y,False))
            elif current[2] == '@' and future == '?':
                self.set_content(current[0]+x,current[1]+y,'+')
                self.set_content(current[0],current[1],' ')
                if save: self.queue.put((x,y,False))
            elif current[2] == '+' and future == ' ':
                self.set_content(current[0]+x,current[1]+y,'@')
                self.set_content(current[0],current[1],'?')
                if save: self.queue.put((x,y,False))
            elif current[2] == '+' and future == '?':
                self.set_content(current[0]+x,current[1]+y,'+')
                self.set_content(current[0],current[1],'?')
                if save: self.queue.put((x,y,False))
        elif self.can_push(x,y):
            current = self.worker()
            future = self.next(x,y)
            future_box = self.next(x+x,y+y)
            if current[2] == '@' and future == '$' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'@')
                if save: self.queue.put((x,y,True))
            elif current[2] == '@' and future == '$' and future_box == '?':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'@')
                if save: self.queue.put((x,y,True))
            elif current[2] == '@' and future == '*' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            elif current[2] == '@' and future == '*' and future_box == '?':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],' ')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            if current[2] == '+' and future == '$' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'?')
                self.set_content(current[0]+x,current[1]+y,'@')
                if save: self.queue.put((x,y,True))
            elif current[2] == '+' and future == '$' and future_box == '?':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'?')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            elif current[2] == '+' and future == '*' and future_box == ' ':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'?')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))
            elif current[2] == '+' and future == '*' and future_box == '?':
                self.move_box(current[0]+x,current[1]+y,x,y)
                self.set_content(current[0],current[1],'?')
                self.set_content(current[0]+x,current[1]+y,'+')
                if save: self.queue.put((x,y,True))

def print_game(matrix, screen):
    save_matrix(matrix)
    screen.fill(background)
    x = 0
    y = 0
    new_size = int(original_size * scale_factor)  # Get updated tile size

    for row in matrix:
        for char in row:
            if char == ' ':
                screen.blit(floor, (x, y))
            elif char == '#':
                screen.blit(wall, (x, y))
            elif char == '@':
                screen.blit(worker, (x, y))
            elif char == '?':
                screen.blit(docker, (x, y))
            elif char == '*':
                screen.blit(box_docked, (x, y))
            elif char == '$':
                screen.blit(box, (x, y))
            elif char == '+':
                screen.blit(worker_docked, (x, y))
            x += new_size  # Move x position by scaled size
        x = 0
        y += new_size  # Move y position by scaled size


def display_box(screen, message):
    fontobject = pygame.font.Font(None, 18)
    pygame.draw.rect(screen, (0, 0, 0),
                     ((screen.get_width() / 2) - 100,
                      (screen.get_height() / 2) - 10,
                      200, 20), 0)
    pygame.draw.rect(screen, (255, 255, 255),
                     ((screen.get_width() / 2) - 102,
                      (screen.get_height() / 2) - 12,
                      204, 24), 1)
    if message:
        screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                    ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
    pygame.display.flip()

def display_end(screen):
    message = "Level Completed"
    fontobject = pygame.font.Font(None, 18)
    pygame.draw.rect(screen, (0, 0, 0),
                     ((screen.get_width() / 2) - 100,
                      (screen.get_height() / 2) - 10,
                      200, 20), 0)
    pygame.draw.rect(screen, (255, 255, 255),
                     ((screen.get_width() / 2) - 102,
                      (screen.get_height() / 2) - 12,
                      204, 24), 1)
    screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
    pygame.display.flip()

def get_key():
    while True:
        event = pygame.event.poll()
        if event.type == pygame.KEYDOWN:
            return event.key

# Load images and initialize pygame
wall = pygame.image.load('games/boxxel/images/wall.png')
floor = pygame.image.load('games/boxxel/images/floor.png')
box = pygame.image.load('games/boxxel/images/box.png')
box_docked = pygame.image.load('games/boxxel/images/box_docked.png')
worker = pygame.image.load('games/boxxel/images/worker.png')
worker_docked = pygame.image.load('games/boxxel/images/worker_dock.png')
docker = pygame.image.load('games/boxxel/images/dock.png')
background = (255, 226, 191)
pygame.init()

# Start game from level 1 and auto advance after completion
level = 1
levels_filename = 'games/boxxel/levels'

while True:
    print("Starting Level " + str(level))
    box_game = game(levels_filename, level)
    size = box_game.load_size()
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    clock = pygame.time.Clock()
    level_completed = False

    while not level_completed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    box_game.move(0, -1, True)
                elif event.key == pygame.K_DOWN:
                    box_game.move(0, 1, True)
                elif event.key == pygame.K_LEFT:
                    box_game.move(-1, 0, True)
                elif event.key == pygame.K_RIGHT:
                    box_game.move(1, 0, True)
                elif event.key == pygame.K_q:
                    sys.exit(0)
                elif event.key == pygame.K_d:
                    box_game.unmove()
                elif event.key == pygame.K_r:
                    box_game = game(levels_filename, level)
            elif event.type == pygame.VIDEORESIZE:
                # Resize the window and update the display
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

                # Calculate scale factor based on the new window width
                scale_factor = event.w / size[0]  # Scale based on width change

                # Update the game elements' sizes
                scale_images()

        if box_game.is_completed():
            print_game(box_game.get_matrix(), screen)  # Ensure the last move is displayed
            pygame.display.update()  # Force screen refresh
            pygame.time.delay(500)  # Small delay to show final state before transition
            
            display_end(screen)  # Show "Level Completed" message
            pygame.display.update()
            pygame.time.delay(2000)  # Wait for 2 seconds before switching levels
            
            level_completed = True


        print_game(box_game.get_matrix(), screen)
        pygame.display.update()
        clock.tick(10)  # Limit to 10 FPS

    level += 1
    # If the level number exceeds the maximum, end the game.
    if level > 52:
        print("Congratulations! All levels completed.")
        sys.exit(0)
