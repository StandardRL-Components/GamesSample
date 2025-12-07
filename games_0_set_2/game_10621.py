import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:44:37.226607
# Source Brief: brief_00621.md
# Brief Index: 621
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Mushroom Maze Environment: A puzzle-platformer where the agent navigates a
    bioluminescent maze. The agent must flip gravity and use colored spores collected
    from mushrooms to open corresponding doors, with the ultimate goal of reaching
    the exit portal before time runs out or falling into a chasm.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Navigate a bioluminescent maze, flip gravity, and use colored spores to open doors. "
        "Reach the exit portal before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press shift to flip gravity and space to shoot a collected spore."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 40
        self.GRID_W, self.GRID_H = self.WIDTH // self.CELL_SIZE, self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 5000
        self.render_mode = render_mode

        # Colors
        self.COLOR_BG_TOP = (15, 5, 30)
        self.COLOR_BG_BOTTOM = (30, 10, 50)
        self.COLOR_WALL = (60, 80, 160)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 60)
        self.SPORE_COLORS = {
            "R": (255, 50, 50),
            "G": (50, 255, 50),
            "B": (50, 100, 255)
        }
        self.COLOR_DOOR_CLOSED = (200, 40, 40)
        self.COLOR_DOOR_OPEN = (40, 200, 40)
        self.COLOR_PORTAL = (240, 240, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CHASM = (10, 0, 20)

        # Physics
        self.GRAVITY = 0.4
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.90
        self.PLAYER_MAX_SPEED = 5
        self.SPORE_SPEED = 7

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 50)
        self._bg_surface = self._create_gradient_background()

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.steps_remaining = 0
        self.reward_this_step = 0
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_radius = self.CELL_SIZE * 0.3
        self.player_spore_type = None
        self.player_last_move_dir = pygame.Vector2(0, 1)
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.gravity_dir = 1
        self.maze = []
        self.path_cells = []
        self.doors = []
        self.mushrooms = []
        self.chasms = []
        self.portal_pos = None
        self.portal_active = False
        
        self.spores = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.steps_remaining = self.MAX_STEPS
        self.reward_this_step = 0

        self.player_vel.update(0, 0)
        self.player_spore_type = None
        self.player_last_move_dir.update(0, 1)
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.gravity_dir = 1
        
        self._generate_solvable_maze()

        self.portal_active = False
        
        self.spores.clear()
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = -0.01 # Time penalty
        
        self._handle_input(action)
        self._update_player()
        self._update_spores()
        self._update_particles()
        
        self._check_interactions()

        self.steps += 1
        self.steps_remaining -= 1
        self.score += self.reward_this_step
        
        terminated = self.game_over or self.steps_remaining <= 0
        truncated = False
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Game Logic ---

    def _handle_input(self, action):
        movement, space_input, shift_input = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement ---
        accel = pygame.Vector2(0, 0)
        if movement == 1: accel.y = -1 # Up
        elif movement == 2: accel.y = 1 # Down
        elif movement == 3: accel.x = -1 # Left
        elif movement == 4: accel.x = 1 # Right
        
        if accel.length() > 0:
            self.player_last_move_dir = accel.normalize()
            self.player_vel += accel * self.PLAYER_ACCEL

        # --- Gravity Flip (Shift) ---
        if shift_input and not self.prev_shift_held:
            self.gravity_dir *= -1
            # SFX: whoosh_gravity_flip.wav
            for _ in range(20):
                p_angle = random.uniform(0, 2 * math.pi)
                p_speed = random.uniform(1, 3)
                p_vel = pygame.Vector2(math.cos(p_angle), math.sin(p_angle)) * p_speed
                self.particles.append({
                    'pos': pygame.Vector2(self.player_pos), 'vel': p_vel,
                    'life': 20, 'max_life': 20, 'radius': random.randint(2, 4),
                    'color': self.COLOR_PLAYER_GLOW
                })
        self.prev_shift_held = shift_input

        # --- Shoot Spore (Space) ---
        if space_input and not self.prev_space_held and self.player_spore_type:
            # SFX: shoot_spore.wav
            spore_dir = pygame.Vector2(self.player_last_move_dir)
            if spore_dir.length_squared() == 0:
                spore_dir.y = float(self.gravity_dir)
            
            self.spores.append({
                'pos': pygame.Vector2(self.player_pos),
                'vel': spore_dir.normalize() * self.SPORE_SPEED,
                'type': self.player_spore_type,
                'life': 120, # 4 seconds
                'trail': deque(maxlen=10)
            })
            self.player_spore_type = None # Consume spore
        self.prev_space_held = space_input

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.gravity_dir * self.GRAVITY
        
        # Apply friction
        self.player_vel *= self.PLAYER_FRICTION
        
        # Limit speed
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)

        # Move and collide
        self.player_pos.x += self.player_vel.x
        self._handle_collisions('x')
        self.player_pos.y += self.player_vel.y
        self._handle_collisions('y')

        # Check for chasm fall
        player_gx, player_gy = int(self.player_pos.x / self.CELL_SIZE), int(self.player_pos.y / self.CELL_SIZE)
        if (player_gx, player_gy) in self.chasms:
            if (self.gravity_dir == 1 and self.player_vel.y > 0) or \
               (self.gravity_dir == -1 and self.player_vel.y < 0):
                self.reward_this_step -= 10
                self.game_over = True
                # SFX: player_fall.wav

    def _handle_collisions(self, axis):
        player_rect = pygame.Rect(self.player_pos.x - self.player_radius, self.player_pos.y - self.player_radius, self.player_radius*2, self.player_radius*2)
        
        # Check for door collisions
        for door in self.doors:
            if not door['is_open']:
                door_rect = pygame.Rect(door['pos'].x, door['pos'].y, self.CELL_SIZE, self.CELL_SIZE)
                if player_rect.colliderect(door_rect):
                    if axis == 'x':
                        if self.player_vel.x > 0: self.player_pos.x = door_rect.left - self.player_radius
                        else: self.player_pos.x = door_rect.right + self.player_radius
                        self.player_vel.x = 0
                    if axis == 'y':
                        if self.player_vel.y > 0: self.player_pos.y = door_rect.top - self.player_radius
                        else: self.player_pos.y = door_rect.bottom + self.player_radius
                        self.player_vel.y = 0

        # Maze wall collisions
        gx, gy = int(self.player_pos.x / self.CELL_SIZE), int(self.player_pos.y / self.CELL_SIZE)
        if not (0 <= gx < self.GRID_W and 0 <= gy < self.GRID_H): return

        cell_walls = self.maze[gy][gx]
        if axis == 'x':
            if self.player_vel.x > 0 and (cell_walls & 0b0100): # Moving right, hit east wall
                self.player_pos.x = (gx + 1) * self.CELL_SIZE - self.player_radius
                self.player_vel.x = 0
            elif self.player_vel.x < 0 and (cell_walls & 0b0001): # Moving left, hit west wall
                self.player_pos.x = gx * self.CELL_SIZE + self.player_radius
                self.player_vel.x = 0
        if axis == 'y':
            if self.player_vel.y > 0 and (cell_walls & 0b0010): # Moving down, hit south wall
                self.player_pos.y = (gy + 1) * self.CELL_SIZE - self.player_radius
                self.player_vel.y = 0
            elif self.player_vel.y < 0 and (cell_walls & 0b1000): # Moving up, hit north wall
                self.player_pos.y = gy * self.CELL_SIZE + self.player_radius
                self.player_vel.y = 0

    def _update_spores(self):
        for spore in self.spores[:]:
            spore['life'] -= 1
            spore['trail'].append(pygame.Vector2(spore['pos']))
            spore['pos'] += spore['vel']
            
            gx, gy = int(spore['pos'].x / self.CELL_SIZE), int(spore['pos'].y / self.CELL_SIZE)

            if not (0 <= gx < self.GRID_W and 0 <= gy < self.GRID_H) or spore['life'] <= 0:
                self.spores.remove(spore)
                continue

            # Spore-wall collision
            cell_walls = self.maze[gy][gx]
            if (spore['vel'].x > 0 and (cell_walls & 0b0100)) or \
               (spore['vel'].x < 0 and (cell_walls & 0b0001)) or \
               (spore['vel'].y > 0 and (cell_walls & 0b0010)) or \
               (spore['vel'].y < 0 and (cell_walls & 0b1000)):
                self.spores.remove(spore)
                # SFX: spore_hit_wall.wav
                continue

            # Spore-door collision
            for door in self.doors:
                if not door['is_open']:
                    door_rect = pygame.Rect(door['pos'].x, door['pos'].y, self.CELL_SIZE, self.CELL_SIZE)
                    if door_rect.collidepoint(spore['pos']):
                        door['hit_sequence'].append(spore['type'])
                        if len(door['hit_sequence']) > len(door['required_sequence']):
                            door['hit_sequence'].pop(0)
                        
                        if door['hit_sequence'] == door['required_sequence']:
                            door['is_open'] = True
                            self.reward_this_step += 1.0
                            # SFX: door_open.wav
                        else:
                            # SFX: door_hit_wrong.wav
                            pass
                        
                        if spore in self.spores: self.spores.remove(spore)
                        break

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            p['pos'] += p['vel']
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_interactions(self):
        # Player and mushrooms
        for mushroom in self.mushrooms:
            dist = self.player_pos.distance_to(mushroom['pos'])
            if dist < self.CELL_SIZE * 0.6:
                if self.player_spore_type != mushroom['type']:
                    self.player_spore_type = mushroom['type']
                    self.reward_this_step += 0.1
                    # SFX: collect_spore.wav

        # Check for win condition
        if not self.portal_active and all(d['is_open'] for d in self.doors):
            self.portal_active = True
        
        if self.portal_active:
            dist = self.player_pos.distance_to(self.portal_pos)
            if dist < self.CELL_SIZE * 0.5:
                self.reward_this_step += 100
                self.game_over = True
                # SFX: win_level.wav

    # --- Maze and Object Generation ---

    def _generate_solvable_maze(self):
        # 1. Generate a perfect maze using recursive backtracking
        self.maze = np.full((self.GRID_H, self.GRID_W), 0b1111, dtype=np.uint8)
        visited = np.zeros((self.GRID_H, self.GRID_W), dtype=bool)
        
        def carve(x, y):
            visited[y, x] = True
            directions = [(0, -1, 0b1000, 0b0010), (0, 1, 0b0010, 0b1000), 
                          (-1, 0, 0b0001, 0b0100), (1, 0, 0b0100, 0b0001)] # (dx, dy, my_wall, their_wall)
            self.np_random.shuffle(directions)
            for dx, dy, my_wall, their_wall in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and not visited[ny, nx]:
                    self.maze[y, x] &= ~my_wall
                    self.maze[ny, nx] &= ~their_wall
                    carve(nx, ny)
        
        carve(self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))

        self.path_cells = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]

        # 2. Find start, end, and path
        start_pos_grid_idx = self.np_random.integers(len(self.path_cells))
        start_pos_grid = self.path_cells[start_pos_grid_idx]
        self.player_pos = pygame.Vector2(start_pos_grid[0] * self.CELL_SIZE + self.CELL_SIZE / 2, 
                                         start_pos_grid[1] * self.CELL_SIZE + self.CELL_SIZE / 2)

        q = deque([(start_pos_grid, [start_pos_grid])])
        visited_bfs = {start_pos_grid}
        longest_path = []
        while q:
            (x, y), path = q.popleft()
            if len(path) > len(longest_path):
                longest_path = path
            
            walls = self.maze[y, x]
            neighbors = [(x, y-1, 0b1000), (x, y+1, 0b0010), (x-1, y, 0b0001), (x+1, y, 0b0100)]
            for nx, ny, wall in neighbors:
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and not (walls & wall) and (nx, ny) not in visited_bfs:
                    visited_bfs.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))

        exit_pos_grid = longest_path[-1]
        main_path = longest_path
        
        self.portal_pos = pygame.Vector2(exit_pos_grid[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
                                         exit_pos_grid[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        # 3. Place doors and mushrooms
        self.doors = []
        self.mushrooms = []
        spore_types = list(self.SPORE_COLORS.keys())
        self.np_random.shuffle(spore_types)
        
        door_indices = [len(main_path) // 4, len(main_path) // 2, 3 * len(main_path) // 4]
        required_sequences = [
            [spore_types[0]],
            [spore_types[0], spore_types[1]],
            [spore_types[0], spore_types[1], spore_types[2]]
        ]

        # Place doors
        for i in range(len(door_indices)):
            if door_indices[i] < len(main_path):
                door_gx, door_gy = main_path[door_indices[i]]
                self.doors.append({
                    'pos': pygame.Vector2(door_gx * self.CELL_SIZE, door_gy * self.CELL_SIZE),
                    'is_open': False, 'required_sequence': required_sequences[i], 'hit_sequence': []
                })

        # Place mushrooms
        non_path_cells = list(set(self.path_cells) - set(main_path))
        self.np_random.shuffle(non_path_cells)
        for i in range(len(spore_types)):
            if non_path_cells:
                mush_gx, mush_gy = non_path_cells.pop()
                self.mushrooms.append({
                    'pos': pygame.Vector2(mush_gx * self.CELL_SIZE + self.CELL_SIZE/2, mush_gy * self.CELL_SIZE + self.CELL_SIZE/2),
                    'type': spore_types[i]
                })

        # 4. Place chasms in dead ends not on the main path
        self.chasms = []
        dead_ends = [c for c in self.path_cells if bin(self.maze[c[1],c[0]]).count('0') == 1 and c not in main_path]
        num_chasms = min(len(dead_ends), 5)
        if dead_ends and num_chasms > 0:
            indices = self.np_random.choice(len(dead_ends), num_chasms, replace=False)
            self.chasms = [dead_ends[i] for i in indices]

    # --- Rendering ---

    def _get_observation(self):
        self.screen.blit(self._bg_surface, (0, 0))
        
        self._render_chasms()
        self._render_maze()
        self._render_doors()
        self._render_mushrooms()
        if self.portal_active: self._render_portal()
        self._render_spores()
        self._render_particles()
        self._render_player()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            color = [
                self.COLOR_BG_TOP[i] + (self.COLOR_BG_BOTTOM[i] - self.COLOR_BG_TOP[i]) * y / self.HEIGHT
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _render_maze(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                walls = self.maze[y, x]
                px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
                if walls & 0b1000: pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), 2) # N
                if walls & 0b0010: pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2) # S
                if walls & 0b0001: pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), 2) # W
                if walls & 0b0100: pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2) # E

    def _render_chasms(self):
        for x, y in self.chasms:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CHASM, rect)

    def _render_doors(self):
        for door in self.doors:
            color = self.COLOR_DOOR_OPEN if door['is_open'] else self.COLOR_DOOR_CLOSED
            rect = pygame.Rect(door['pos'].x, door['pos'].y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, 3, border_radius=4)

    def _render_mushrooms(self):
        for mush in self.mushrooms:
            pulsation = (math.sin(self.steps * 0.1 + mush['pos'].x) + 1) / 2
            radius = self.CELL_SIZE * 0.2 + pulsation * 4
            color = self.SPORE_COLORS[mush['type']]
            pos_int = (int(mush['pos'].x), int(mush['pos'].y))
            glow_color = (*color, 30 + int(pulsation * 30))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(radius * 1.8), glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius), color)

    def _render_portal(self):
        pos_int = (int(self.portal_pos.x), int(self.portal_pos.y))
        for i in range(5):
            angle = (self.steps * 0.02 + i * math.pi * 2 / 5)
            radius_factor = (math.sin(self.steps * 0.05 + i) + 1.5) / 2.5
            radius = int(self.CELL_SIZE * 0.4 * radius_factor)
            offset_x = int(math.cos(angle) * self.CELL_SIZE * 0.2)
            offset_y = int(math.sin(angle) * self.CELL_SIZE * 0.2)
            color = (*self.COLOR_PORTAL, 100)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0] + offset_x, pos_int[1] + offset_y, radius, color)

    def _render_spores(self):
        for spore in self.spores:
            color = self.SPORE_COLORS[spore['type']]
            # Trail
            if len(spore['trail']) > 1:
                for i, p in enumerate(spore['trail']):
                    alpha = int(200 * (i / len(spore['trail'])))
                    trail_color = (*color, alpha)
                    pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), 3, trail_color)
            # Head
            pos_int = (int(spore['pos'].x), int(spore['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 5, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 5, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'][:3], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

    def _render_player(self):
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.player_radius * 2.0), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.player_radius * 1.5), self.COLOR_PLAYER_GLOW)
        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.player_radius), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(self.player_radius), self.COLOR_PLAYER)
        # Current spore indicator
        if self.player_spore_type:
            spore_color = self.SPORE_COLORS[self.player_spore_type]
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.player_radius * 0.5), spore_color)

    def _render_ui(self):
        # Timer
        timer_text = self.font_ui.render(f"Time: {self.steps_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 5))
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 30))
        # Gravity Arrow
        arrow_points = []
        if self.gravity_dir == 1: # Down
            arrow_points = [(20, 10), (40, 10), (30, 25)]
        else: # Up
            arrow_points = [(30, 5), (20, 20), (40, 20)]
        pygame.draw.polygon(self.screen, self.COLOR_TEXT, arrow_points)
        # Spore indicator
        if self.player_spore_type:
            text = self.font_ui.render("Spore:", True, self.COLOR_TEXT)
            self.screen.blit(text, (10, self.HEIGHT - 35))
            pygame.draw.circle(self.screen, self.SPORE_COLORS[self.player_spore_type], (80, self.HEIGHT - 22), 10)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        status_text = "VICTORY!" if self.portal_active and self.player_pos.distance_to(self.portal_pos) < self.CELL_SIZE * 0.5 else "GAME OVER"
        text_surf = self.font_big.render(status_text, True, self.COLOR_PLAYER)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "steps_remaining": self.steps_remaining}

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows direct execution of the file for testing purposes.
    # It sets up a pygame window and allows for human control of the agent.
    
    # Re-enable display for interactive mode
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Mushroom Maze")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    done = True

        clock.tick(env.metadata['render_fps'])

    env.close()