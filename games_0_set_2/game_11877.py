import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:44:17.187022
# Source Brief: brief_01877.md
# Brief Index: 1877
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
    A Gymnasium environment where the agent guides a light beam using mirrors.

    **Objective:** Collect all energy cells by reflecting a light beam off strategically
    placed mirrors before the timer runs out.

    **Visuals:** Minimalist, geometric style with glowing neon effects. The game prioritizes
    visual feedback and a satisfying "game feel".

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `actions[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right. Moves the placement cursor.
    - `actions[1]` (Place/Fire): 0=Released, 1=Pressed. Places a mirror if the cell is empty,
      then fires the light beam.
    - `actions[2]` (Rotate): 0=Released, 1=Pressed. Rotates the mirror at the cursor's
      location 90 degrees clockwise.

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - +5 for each energy cell collected.
    - +2.5 bonus for each subsequent cell in a single chain.
    - +0.1 for each successful reflection.
    - +100 bonus for winning the game (collecting all cells).
    - -100 penalty for losing (timer runs out).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a light beam using mirrors to collect all energy cells. "
        "Strategically place and rotate mirrors to reflect the beam before the timer runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place a mirror and fire the beam. "
        "Press shift to rotate a mirror."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 900  # 30 seconds at 30 FPS
    NUM_CELLS = 5
    NUM_OBSTACLES = 10
    BEAM_ANIM_STEPS = 15 # Steps for beam to travel its full path

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_OBSTACLE = (120, 30, 30)
    COLOR_OBSTACLE_GLOW = (200, 50, 50)
    COLOR_CELL = (50, 200, 50)
    COLOR_CELL_GLOW = (150, 255, 150)
    COLOR_MIRROR = (100, 180, 255)
    COLOR_BEAM = (255, 255, 100)
    COLOR_BEAM_GLOW = (255, 255, 180)
    COLOR_TEXT = (220, 220, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.mirrors = {}
        self.energy_cells = set()
        self.obstacles = set()
        self.beam_active = False
        self.beam_path = []
        self.beam_anim_progress = 0.0
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.info_text_queue = deque(maxlen=5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.mirrors = {}
        self.beam_active = False
        self.beam_path = []
        self.beam_anim_progress = 0.0
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.info_text_queue.clear()
        
        # Procedural Level Generation
        self.energy_cells = set()
        self.obstacles = set()
        
        # Beam starts at left-center
        self.beam_origin_pos = (0, self.GRID_H // 2)
        self.beam_origin_dir = (1, 0)

        # Generate non-overlapping positions
        possible_positions = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))
        possible_positions.discard(tuple(self.cursor_pos))
        possible_positions.discard(self.beam_origin_pos)

        # Place energy cells
        for _ in range(self.NUM_CELLS):
            if not possible_positions: break
            pos = random.choice(list(possible_positions))
            self.energy_cells.add(pos)
            possible_positions.remove(pos)

        # Place obstacles
        for _ in range(self.NUM_OBSTACLES):
            if not possible_positions: break
            pos = random.choice(list(possible_positions))
            self.obstacles.add(pos)
            possible_positions.remove(pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # Update game state only if beam is not animating
        if not self.beam_active:
            # --- Handle Actions ---
            self._handle_movement(movement)
            
            space_pressed = space_held and not self.last_space_held
            shift_pressed = shift_held and not self.last_shift_held

            if shift_pressed:
                self._rotate_mirror() # sfx: rotate_mirror.wav
            
            if space_pressed:
                self._place_mirror()
                reward += self._fire_beam() # sfx: fire_beam.wav

        # --- Update Game Systems ---
        self._update_beam_animation()
        self._update_particles()
        
        self.steps += 1
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Check Termination Conditions ---
        terminated = False
        if len(self.energy_cells) == 0:
            reward += 100.0
            self._add_info_text("LEVEL CLEAR!", (100, 255, 100))
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 100.0
            self._add_info_text("TIME OUT!", (255, 100, 100))
            terminated = True
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

    def _rotate_mirror(self):
        cursor_tuple = tuple(self.cursor_pos)
        if cursor_tuple in self.mirrors:
            self.mirrors[cursor_tuple] = 1 - self.mirrors[cursor_tuple] # Flip orientation

    def _place_mirror(self):
        cursor_tuple = tuple(self.cursor_pos)
        if cursor_tuple not in self.mirrors and cursor_tuple not in self.obstacles and cursor_tuple not in self.energy_cells:
            self.mirrors[cursor_tuple] = 0 # Default orientation: /

    def _fire_beam(self):
        self.beam_active = True
        self.beam_anim_progress = 0.0
        
        path, cells_hit, reflections = self._calculate_beam_path()
        self.beam_path = path

        # Calculate reward for this shot
        shot_reward = 0.0
        shot_reward += reflections * 0.1
        
        if cells_hit:
            self.score += len(cells_hit)
            shot_reward += len(cells_hit) * 5.0
            if len(cells_hit) > 1:
                bonus = (len(cells_hit) - 1) * 2.5
                shot_reward += bonus
                self._add_info_text(f"CHAIN x{len(cells_hit)}! +{int(bonus)}", self.COLOR_CELL_GLOW)
            
            for cell_pos in cells_hit:
                if cell_pos in self.energy_cells:
                    self.energy_cells.remove(cell_pos)
                    # sfx: collect_cell.wav
                    self._spawn_particles(self._grid_to_pixel(cell_pos), 20, self.COLOR_CELL_GLOW)
        
        if not cells_hit and reflections == 0 and len(path) <= 2:
            pass # No reward for firing into nothing
        elif not cells_hit:
            # sfx: beam_hit_wall.wav
            self._spawn_particles(path[-1], 10, self.COLOR_BEAM_GLOW)
        
        return shot_reward

    def _calculate_beam_path(self):
        pos = list(self.beam_origin_pos)
        direction = list(self.beam_origin_dir)
        path = [self._grid_to_pixel(pos, center=True)]
        cells_hit = []
        reflections = 0

        for _ in range(self.GRID_W * self.GRID_H): # Limit path length
            pos[0] += direction[0]
            pos[1] += direction[1]

            if not (0 <= pos[0] < self.GRID_W and 0 <= pos[1] < self.GRID_H):
                path.append(self._grid_to_pixel(pos, center=True))
                break

            path.append(self._grid_to_pixel(pos, center=True))
            pos_tuple = tuple(pos)
            
            if pos_tuple in self.obstacles:
                # sfx: beam_hit_obstacle.wav
                break
            
            if pos_tuple in self.energy_cells and pos_tuple not in cells_hit:
                cells_hit.append(pos_tuple)

            if pos_tuple in self.mirrors:
                reflections += 1
                orientation = self.mirrors[pos_tuple] # 0 for /, 1 for \
                dx, dy = direction
                if orientation == 0: # '/'
                    direction = [-dy, -dx]
                else: # '\'
                    direction = [dy, dx]
        
        return path, cells_hit, reflections

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_obstacles()
        self._render_energy_cells()
        self._render_mirrors()
        self._render_cursor()
        self._render_beam()
        self._render_particles()

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Beam origin indicator
        ox, oy = self._grid_to_pixel(self.beam_origin_pos, center=True)
        pygame.draw.circle(self.screen, self.COLOR_BEAM, (ox, oy), 8)
        pygame.draw.circle(self.screen, self.COLOR_GRID, (ox, oy), 8, 2)


    def _render_obstacles(self):
        for x, y in self.obstacles:
            px, py = self._grid_to_pixel((x, y))
            rect = pygame.Rect(px + 5, py + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, rect, width=2, border_radius=3)

    def _render_energy_cells(self):
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        radius = int(self.CELL_SIZE * 0.25 + pulse * 3)
        glow_radius = int(radius * (1.5 + pulse * 0.5))
        
        for x, y in self.energy_cells:
            px, py = self._grid_to_pixel((x, y), center=True)
            
            # Glow effect
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_CELL_GLOW, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_CELL)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_CELL_GLOW)

    def _render_mirrors(self):
        for (x, y), orientation in self.mirrors.items():
            px, py = self._grid_to_pixel((x, y))
            margin = 8
            if orientation == 0: # '/'
                start = (px + margin, py + self.CELL_SIZE - margin)
                end = (px + self.CELL_SIZE - margin, py + margin)
            else: # '\'
                start = (px + margin, py + margin)
                end = (px + self.CELL_SIZE - margin, py + self.CELL_SIZE - margin)
            pygame.draw.line(self.screen, self.COLOR_MIRROR, start, end, 4)

    def _render_cursor(self):
        x, y = self.cursor_pos
        px, py = self._grid_to_pixel((x, y))
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=3)
        
    def _render_beam(self):
        if not self.beam_active or len(self.beam_path) < 2:
            return

        total_length = sum(math.hypot(self.beam_path[i+1][0] - self.beam_path[i][0], self.beam_path[i+1][1] - self.beam_path[i][1]) for i in range(len(self.beam_path)-1))
        draw_length = total_length * self.beam_anim_progress
        
        current_length = 0
        for i in range(len(self.beam_path) - 1):
            p1 = self.beam_path[i]
            p2 = self.beam_path[i+1]
            segment_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

            if current_length + segment_length > draw_length:
                ratio = (draw_length - current_length) / segment_length
                p_end = (p1[0] + (p2[0] - p1[0]) * ratio, p1[1] + (p2[1] - p1[1]) * ratio)
                pygame.draw.line(self.screen, self.COLOR_BEAM_GLOW, p1, p_end, 6)
                pygame.draw.line(self.screen, self.COLOR_BEAM, p1, p_end, 2)
                # Spawn particles at the head of the beam
                if self.steps % 2 == 0:
                    self._spawn_particles(p_end, 1, self.COLOR_BEAM, speed_scale=0.5)
                break
            else:
                pygame.draw.line(self.screen, self.COLOR_BEAM_GLOW, p1, p2, 6)
                pygame.draw.line(self.screen, self.COLOR_BEAM, p1, p2, 2)
                current_length += segment_length

    def _update_beam_animation(self):
        if self.beam_active:
            self.beam_anim_progress += 1.0 / self.BEAM_ANIM_STEPS
            if self.beam_anim_progress >= 1.0:
                self.beam_active = False
                self.beam_anim_progress = 0.0

    def _spawn_particles(self, pos, count, color, speed_scale=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_scale
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(15, 30)
            self.particles.append([list(pos), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _render_particles(self):
        for pos, vel, lifetime, color in self.particles:
            alpha = int(255 * (lifetime / 30.0))
            radius = int(lifetime / 10)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, (*color, alpha))

    def _render_ui(self):
        # Timer Bar
        time_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        bar_width = int(self.WIDTH * time_ratio)
        bar_color = (int(200 * (1 - time_ratio)), int(200 * time_ratio), 50)
        pygame.draw.rect(self.screen, bar_color, (0, 0, bar_width, 5))

        # Score Text
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Cells Remaining
        cells_surf = self.font_small.render(f"CELLS: {len(self.energy_cells)}/{self.NUM_CELLS}", True, self.COLOR_TEXT)
        self.screen.blit(cells_surf, (self.WIDTH // 2 - cells_surf.get_width() // 2, 10))
        
        # Timer Text
        time_left = (self.MAX_STEPS - self.steps) / 30.0
        timer_surf = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Info Text
        for i, (text, color, lifetime) in enumerate(list(self.info_text_queue)):
            if self.steps > lifetime:
                continue
            alpha = int(255 * (lifetime - self.steps) / 50)
            if alpha > 0:
                info_surf = self.font_large.render(text, True, color)
                info_surf.set_alpha(alpha)
                pos_y = self.HEIGHT - 50 - i * 35
                self.screen.blit(info_surf, (self.WIDTH // 2 - info_surf.get_width() // 2, pos_y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cells_remaining": len(self.energy_cells),
            "mirrors_placed": len(self.mirrors)
        }

    def _grid_to_pixel(self, grid_pos, center=False):
        x, y = grid_pos
        px = x * self.CELL_SIZE
        py = y * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return int(px), int(py)

    def _add_info_text(self, text, color):
        self.info_text_queue.append((text, color, self.steps + 50)) # Display for 50 steps

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For local testing, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Light Grid")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # --- Action mapping for human play ---
    # action = [movement, space, shift]
    action = [0, 0, 0] 

    print("\n--- Manual Control ---")
    print("Arrows: Move cursor")
    print("Space: Place mirror and fire beam")
    print("Shift: Rotate mirror")
    print("R: Reset environment")
    print("Q: Quit")
    
    # Remove the validation call from the main loop
    # env.validate_implementation() 
    
    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"Environment Reset. Initial Info: {info}")
        
        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        # Buttons
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")

        if terminated or truncated:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {info['score']}, Final Reward: {total_reward:.2f}")
            # Wait for a moment before quitting
            pygame.time.wait(2000)
            done = True

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()