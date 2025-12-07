import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Eat all the pellets to win, but avoid the ghosts!"
    )

    game_description = (
        "A retro arcade maze game. Gobble pellets for points while evading patrolling ghosts. "
        "Eating a large power pellet will temporarily turn the ghosts vulnerable."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (20, 50, 150)
    COLOR_WALL_TOP = (40, 80, 220)
    COLOR_PELLET = (255, 255, 0)
    COLOR_POWER_PELLET = (255, 255, 255)
    COLOR_PLAYER = (255, 255, 0)
    GHOST_COLORS = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]
    COLOR_GHOST_VULNERABLE = (50, 50, 255)
    COLOR_GHOST_VULNERABLE_FLASH = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)

    # Maze Dimensions
    MAZE_WIDTH = 20
    MAZE_HEIGHT = 11
    CELL_SIZE = 32
    WALL_THICKNESS = 4

    # Game Parameters
    TOTAL_PELLETS = 100
    POWER_PELLET_COUNT = 4
    MAX_STEPS = 1000
    PLAYER_SPEED = 0.125  # tiles per step
    GHOST_INITIAL_SPEED = 0.11  # tiles per step
    GHOST_VULNERABLE_SPEED = 0.08
    GHOST_EATEN_SPEED = 0.25
    VULNERABLE_DURATION = 150  # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.game_screen_width = self.MAZE_WIDTH * self.CELL_SIZE
        self.game_screen_height = self.MAZE_HEIGHT * self.CELL_SIZE
        
        self.obs_width = 640
        self.obs_height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.obs_height, self.obs_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        # This surface is for rendering the game area only
        self.screen = pygame.Surface((self.game_screen_width, self.game_screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)

        self.player = None
        self.ghosts = []
        self.maze = None
        self.pellets = set()
        self.power_pellets = set()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.vulnerable_timer = 0
        self.initial_pellet_count = 0

    def _generate_maze(self):
        w, h = self.MAZE_WIDTH, self.MAZE_HEIGHT
        maze = np.ones((h, w), dtype=np.int8)

        def is_valid(x, y):
            return 0 <= x < w and 0 <= y < h

        # Randomized DFS
        stack = deque()
        start_x = self.np_random.integers(1, w - 1)
        start_y = self.np_random.integers(1, h - 1)
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny) and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                # np.random.Generator.choice expects an array-like
                chosen_idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[chosen_idx]
                mx, my = (cx + nx) // 2, (cy + ny) // 2
                maze[ny, nx] = 0
                maze[my, mx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        # Make it less perfect by removing some walls
        for _ in range(int(w * h * 0.15)):
            rx, ry = self.np_random.integers(1, w - 1), self.np_random.integers(1, h - 1)
            if maze[ry, rx] == 1:
                path_neighbors = 0
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if maze[ry + dy, rx + dx] == 0:
                        path_neighbors += 1
                if path_neighbors >= 2:
                    maze[ry, rx] = 0

        return maze

    def _populate_maze(self):
        self.pellets.clear()
        self.power_pellets.clear()

        path_cells = []
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y, x] == 0:
                    path_cells.append((x, y))

        self.np_random.shuffle(path_cells)

        # Place Player
        player_pos = path_cells.pop()
        self.player = Player(player_pos[0], player_pos[1], self.PLAYER_SPEED)

        # Place Ghosts
        self.ghosts = []
        ghost_start_area = [c for c in path_cells if self._dist(c, player_pos) > 5]
        if not ghost_start_area: ghost_start_area = path_cells

        for i in range(4):
            if not ghost_start_area: break
            pos = ghost_start_area.pop(self.np_random.integers(len(ghost_start_area)))
            self.ghosts.append(Ghost(pos[0], pos[1], self.GHOST_INITIAL_SPEED, self.GHOST_COLORS[i], i))

        # Place Pellets
        pellet_candidates = [c for c in path_cells if self._dist(c, player_pos) > 1]
        self.np_random.shuffle(pellet_candidates)

        # Power Pellets
        for _ in range(self.POWER_PELLET_COUNT):
            if not pellet_candidates: break
            self.power_pellets.add(pellet_candidates.pop())

        # Regular Pellets
        num_to_place = min(len(pellet_candidates), self.TOTAL_PELLETS - len(self.power_pellets))
        for i in range(num_to_place):
            self.pellets.add(pellet_candidates[i])
        self.initial_pellet_count = len(self.pellets) + len(self.power_pellets)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.maze = self._generate_maze()
        self._populate_maze()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.vulnerable_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0

        # --- Update Player ---
        self.player.set_next_direction(movement)
        self.player.update(self.maze)
        player_grid_pos = (int(self.player.x + 0.5), int(self.player.y + 0.5))

        # --- Pellet Collision ---
        if player_grid_pos in self.pellets:
            self.pellets.remove(player_grid_pos)
            self.score += 10
            reward += 1
            pellets_eaten = self.initial_pellet_count - (len(self.pellets) + len(self.power_pellets))
            if pellets_eaten > 0 and pellets_eaten % 25 == 0:
                for ghost in self.ghosts:
                    ghost.base_speed += 0.005

        if player_grid_pos in self.power_pellets:
            self.power_pellets.remove(player_grid_pos)
            self.score += 50
            reward += 10
            self.vulnerable_timer = self.VULNERABLE_DURATION
            for ghost in self.ghosts:
                if ghost.state != 'eaten':
                    ghost.state = 'vulnerable'
                    ghost.direction = (ghost.direction[0] * -1, ghost.direction[1] * -1)  # Reverse

        # --- Update Ghosts ---
        if self.vulnerable_timer > 0:
            self.vulnerable_timer -= 1
            if self.vulnerable_timer == 0:
                for ghost in self.ghosts:
                    if ghost.state == 'vulnerable':
                        ghost.state = 'chase'

        for ghost in self.ghosts:
            ghost.update(self.maze, self.player, self.ghosts, self.GHOST_VULNERABLE_SPEED, self.GHOST_EATEN_SPEED, self.np_random)

        # --- Ghost Collision ---
        for ghost in self.ghosts:
            if self._dist((self.player.x, self.player.y), (ghost.x, ghost.y)) < 0.5:
                if ghost.state == 'vulnerable':
                    ghost.state = 'eaten'
                    ghost.target = ghost.start_pos
                    self.score += 200
                    reward += 5
                elif ghost.state != 'eaten':
                    self.game_over = True
                    reward -= 100

        # --- Reward Shaping ---
        chase_ghosts = [g for g in self.ghosts if g.state == 'chase']
        if chase_ghosts:
            nearest_ghost_dist = min(self._dist((self.player.x, self.player.y), (g.x, g.y)) for g in chase_ghosts)
            if movement != 0 and nearest_ghost_dist < 4:
                px, py = self.player.x, self.player.y
                moved_pos = (px + self.player.direction[0], py + self.player.direction[1])
                new_dist = min(self._dist(moved_pos, (g.x, g.y)) for g in chase_ghosts)
                if new_dist > nearest_ghost_dist:
                    reward -= 0.2  # Penalty for moving away from nearby ghost
        
        # --- Termination ---
        terminated = self.game_over
        if not self.pellets and not self.power_pellets:
            terminated = True
            reward += 100  # Win bonus
            self.game_over = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Render the game to the internal screen surface
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        # Create the final observation surface with the required dimensions
        obs_surface = pygame.Surface((self.obs_width, self.obs_height))
        obs_surface.fill(self.COLOR_BG)

        # Blit the game screen onto the center of the observation surface
        render_offset_y = (self.obs_height - self.game_screen_height) // 2
        obs_surface.blit(self.screen, (0, render_offset_y))

        # Convert to numpy array and transpose to (H, W, C)
        arr = pygame.surfarray.array3d(obs_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pellets_left": len(self.pellets) + len(self.power_pellets),
            "vulnerable_timer": self.vulnerable_timer,
        }

    def _render_game(self):
        # Draw Pellets
        for x, y in self.pellets:
            px, py = int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE)
            pygame.draw.circle(self.screen, self.COLOR_PELLET, (px, py), 3)

        # Draw Power Pellets
        is_flashing = (self.steps // 5) % 2 == 0
        pellet_size = 8 if is_flashing else 6
        for x, y in self.power_pellets:
            px, py = int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE)
            pygame.draw.circle(self.screen, self.COLOR_POWER_PELLET, (px, py), pellet_size)

        # Draw Maze
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    if y > 0 and self.maze[y - 1, x] == 0:
                        pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, (rect.x, rect.y, self.CELL_SIZE, self.WALL_THICKNESS))

        # Draw Ghosts
        for ghost in self.ghosts:
            ghost.draw(self.screen, self.CELL_SIZE, self.steps, self.vulnerable_timer, self.COLOR_GHOST_VULNERABLE, self.COLOR_GHOST_VULNERABLE_FLASH)

        # Draw Player
        self.player.draw(self.screen, self.CELL_SIZE, self.steps, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        pellets_left = len(self.pellets) + len(self.power_pellets)
        pellets_text = self.font_ui.render(f"PELLETS: {pellets_left}", True, self.COLOR_UI_TEXT)
        text_rect = pellets_text.get_rect(bottomright=(self.game_screen_width - 10, self.game_screen_height - 10))
        self.screen.blit(pellets_text, text_rect)

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def close(self):
        pygame.quit()


class Player:
    def __init__(self, x, y, speed):
        self.x, self.y = float(x), float(y)
        self.speed = speed
        self.direction = (0, 0)
        self.next_direction = (0, 0)

    def set_next_direction(self, movement_action):
        if movement_action == 1: self.next_direction = (0, -1)  # Up
        elif movement_action == 2: self.next_direction = (0, 1)  # Down
        elif movement_action == 3: self.next_direction = (-1, 0)  # Left
        elif movement_action == 4: self.next_direction = (1, 0)  # Right

    def _is_at_center(self):
        return abs(self.x - round(self.x)) < self.speed and abs(self.y - round(self.y)) < self.speed

    def _can_move(self, direction, maze):
        if direction == (0, 0): return True
        ix, iy = int(self.x + 0.5), int(self.y + 0.5)
        nx, ny = ix + direction[0], iy + direction[1]
        return 0 <= nx < GameEnv.MAZE_WIDTH and 0 <= ny < GameEnv.MAZE_HEIGHT and maze[ny, nx] == 0

    def update(self, maze):
        if self._is_at_center():
            self.x, self.y = round(self.x), round(self.y)
            if self.next_direction != (0, 0) and self._can_move(self.next_direction, maze):
                self.direction = self.next_direction
            if not self._can_move(self.direction, maze):
                self.direction = (0, 0)

        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed

        if self.x < -0.5: self.x = GameEnv.MAZE_WIDTH - 0.51
        if self.x > GameEnv.MAZE_WIDTH - 0.5: self.x = -0.49

    def draw(self, surface, cell_size, steps, color):
        px, py = int((self.x + 0.5) * cell_size), int((self.y + 0.5) * cell_size)
        radius = cell_size // 2 - 4

        angle_offset = (math.sin(steps * 0.5) + 1) / 2 * (math.pi / 4)

        if self.direction == (0, 0):
            pygame.draw.circle(surface, color, (px, py), radius)
            return

        p1 = (px, py)
        angle = math.atan2(self.direction[1], self.direction[0])

        point_list = [p1]
        for i in range(16):
            a = angle + angle_offset + (math.pi * 2 - 2 * angle_offset) * i / 15
            point_list.append((px + radius * math.cos(a), py + radius * math.sin(a)))

        pygame.gfxdraw.aapolygon(surface, [(int(x), int(y)) for x, y in point_list], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(x), int(y)) for x, y in point_list], color)


class Ghost(Player):
    def __init__(self, x, y, speed, color, ghost_id):
        super().__init__(x, y, speed)
        self.start_pos = (x, y)
        self.base_speed = speed
        self.color = color
        self.ghost_id = ghost_id
        self.state = 'chase'  # chase, vulnerable, eaten
        self.target = None
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

    def update(self, maze, player, ghosts, vulnerable_speed, eaten_speed, np_random):
        self.speed = self.base_speed
        if self.state == 'vulnerable':
            self.speed = vulnerable_speed
        elif self.state == 'eaten':
            self.speed = eaten_speed
            if self._dist((self.x, self.y), self.start_pos) < 0.5:
                self.state = 'chase'
                self.x, self.y = self.start_pos

        if self._is_at_center():
            self.x, self.y = round(self.x), round(self.y)
            self._update_target(player, ghosts)
            self._choose_direction(maze, np_random)

        super().update(maze)

    def _update_target(self, player, ghosts):
        if self.state == 'eaten':
            self.target = self.start_pos
            return
        if self.state == 'vulnerable':
            self.target = (-player.x, -player.y)
            return

        if self.ghost_id == 0:  # Red: Direct chase
            self.target = (player.x, player.y)
        elif self.ghost_id == 1:  # Pink: Ambush
            self.target = (player.x + player.direction[0] * 4, player.y + player.direction[1] * 4)
        elif self.ghost_id == 2:  # Cyan: Patrol
            red_ghost = ghosts[0]
            dx, dy = player.x - red_ghost.x, player.y - red_ghost.y
            self.target = (player.x + dx, player.y + dy)
        elif self.ghost_id == 3:  # Orange: Erratic
            if self._dist((self.x, self.y), (player.x, player.y)) > 8:
                self.target = (player.x, player.y)
            else:
                self.target = (0, GameEnv.MAZE_HEIGHT)

    def _choose_direction(self, maze, np_random):
        ix, iy = int(self.x + 0.5), int(self.y + 0.5)
        possible_moves = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if (dx, dy) == (-self.direction[0], -self.direction[1]):
                continue
            if self._can_move((dx, dy), maze):
                possible_moves.append((dx, dy))

        if not possible_moves:
            self.direction = (-self.direction[0], -self.direction[1])
            if not self._can_move(self.direction, maze):
                self.direction = (0, 0)
            return

        if self.target is None:
            idx = np_random.integers(len(possible_moves))
            self.direction = possible_moves[idx]
            return

        best_move = None
        min_dist = float('inf')
        for move in possible_moves:
            nx, ny = ix + move[0], iy + move[1]
            dist = self._dist((nx, ny), self.target)
            if dist < min_dist:
                min_dist = dist
                best_move = move
        
        if best_move:
            self.direction = best_move
        else: # Should not happen if possible_moves is not empty
            idx = np_random.integers(len(possible_moves))
            self.direction = possible_moves[idx]


    def draw(self, surface, cell_size, steps, vulnerable_timer, vulnerable_color, flash_color):
        px, py = int((self.x + 0.5) * cell_size), int((self.y + 0.5) * cell_size)
        w, h = cell_size - 8, cell_size - 8

        body_rect = pygame.Rect(px - w // 2, py - h // 2, w, h)

        color = self.color
        if self.state == 'vulnerable':
            color = flash_color if vulnerable_timer < 60 and (steps // 5) % 2 == 0 else vulnerable_color

        if self.state == 'eaten':
            eye_color = (255, 255, 255)
            eye_l_pos = (px - w // 4, py)
            eye_r_pos = (px + w // 4, py)
            pygame.draw.circle(surface, eye_color, eye_l_pos, 2)
            pygame.draw.circle(surface, eye_color, eye_r_pos, 2)
            return

        pygame.draw.rect(surface, color, (body_rect.x, body_rect.y, w, h / 2))
        pygame.draw.circle(surface, color, (body_rect.centerx, body_rect.y + h / 4), w / 2)

        num_waves = 3
        wave_width = w / num_waves
        for i in range(num_waves):
            cx = body_rect.x + i * wave_width + wave_width / 2
            cy = body_rect.y + h / 2
            pygame.draw.circle(surface, color, (int(cx), int(cy)), int(wave_width / 2))

        eye_color = (255, 255, 255)
        pupil_color = (0, 0, 0)
        eye_l_pos = (px - w // 4, py - h // 8)
        eye_r_pos = (px + w // 4, py - h // 8)
        pygame.draw.circle(surface, eye_color, eye_l_pos, 4)
        pygame.draw.circle(surface, eye_color, eye_r_pos, 4)

        pupil_dx, pupil_dy = self.direction[0] * 2, self.direction[1] * 2
        pygame.draw.circle(surface, pupil_color, (eye_l_pos[0] + pupil_dx, eye_l_pos[1] + pupil_dy), 2)
        pygame.draw.circle(surface, pupil_color, (eye_r_pos[0] + pupil_dx, eye_r_pos[1] + pupil_dy), 2)

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)