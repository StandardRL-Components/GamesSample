import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:57:32.324188
# Source Brief: brief_00730.md
# Brief Index: 730
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Cell:
    def __init__(self, pos, radius=20):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.radius = radius
        self.max_radius = 40
        self.min_radius = 10
        self.color = (50, 200, 255) # Bright Cyan

    def apply_force(self, force):
        self.vel += force

    def update(self, drag, world_size):
        self.pos += self.vel
        self.vel *= drag

        # World wrapping
        self.pos.x %= world_size[0]
        self.pos.y %= world_size[1]

    def grow(self, amount):
        self.radius = min(self.max_radius, self.radius + amount)

    def shrink(self, amount):
        self.radius = max(self.min_radius, self.radius - amount)

class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        return self.lifespan > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide a cluster of cells through a dangerous field of obstacles to reach the safety of the exit. "
        "Strategically grow and shrink your cells to navigate tight spaces."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cell cluster. Press space or shift to cycle which cell is selected. "
        "Hold space to shrink the selected cell and hold shift to grow it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_W, self.SCREEN_H = 640, 400
        self.WORLD_W, self.WORLD_H = 2000, 1500
        self.FPS = 30 # Assumed FPS for smooth interpolation
        self.MAX_STEPS = 1500

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_OBSTACLE = (50, 60, 80)
        self.COLOR_EXIT = (255, 220, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_SELECTED_GLOW = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cells = []
        self.obstacles = []
        self.particles = []
        self.exit_pos = pygame.math.Vector2(0, 0)
        self.exit_radius = 0
        self.camera_pos = pygame.math.Vector2(0, 0)
        self.selected_cell_idx = 0
        self.last_distance_to_exit = 0
        self.obstacle_density = 0.05
        
        # Action state tracking for press vs. hold
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by tests

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_density = 0.05

        # Initialize player cells
        self.cells = []
        start_pos = pygame.math.Vector2(self.WORLD_W / 2, self.WORLD_H / 2)
        for _ in range(3):
            offset = pygame.math.Vector2(
                self.np_random.uniform(-30, 30), self.np_random.uniform(-30, 30)
            )
            self.cells.append(Cell(start_pos + offset, radius=20))

        # Place exit far from start
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(self.WORLD_W * 0.4, self.WORLD_W * 0.45)
        self.exit_pos = start_pos + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * dist
        self.exit_radius = 50

        # Generate obstacles
        self._spawn_obstacles()

        self.particles = []
        self.selected_cell_idx = 0
        
        if self.cells:
            avg_pos = self._get_average_cell_position()
            self.last_distance_to_exit = avg_pos.distance_to(self.exit_pos)
            self.camera_pos = avg_pos - pygame.math.Vector2(self.SCREEN_W / 2, self.SCREEN_H / 2)
        else:
            self.last_distance_to_exit = 0
            self.camera_pos = pygame.math.Vector2(0,0)
            
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if self.cells:
            # Cycle selection on button press
            if space_pressed or shift_pressed:
                # sfx: UI_Select_Sound
                self.selected_cell_idx = (self.selected_cell_idx + 1) % len(self.cells)

            selected_cell = self.cells[self.selected_cell_idx]
            if space_held:
                selected_cell.shrink(0.5) # sfx: Shrink_Sound_Loop
            if shift_held:
                selected_cell.grow(0.5) # sfx: Grow_Sound_Loop

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        move_force = pygame.math.Vector2(0, 0)
        if movement == 1: move_force.y = -1 # Up
        elif movement == 2: move_force.y = 1  # Down
        elif movement == 3: move_force.x = -1 # Left
        elif movement == 4: move_force.x = 1  # Right
        
        if move_force.length() > 0:
            move_force.normalize_ip()
            for cell in self.cells:
                # Bigger cells are slower to push
                cell.apply_force(move_force * (60 / (cell.radius + 10)))

        # --- Game Logic Update ---
        for cell in self.cells:
            cell.update(drag=0.92, world_size=(self.WORLD_W, self.WORLD_H))
            
        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # --- Collision Detection ---
        cells_to_remove = []
        for i, cell in enumerate(self.cells):
            # Obstacle collision
            for obstacle in self.obstacles:
                if cell.pos.distance_to(obstacle['pos']) < cell.radius + obstacle['radius']:
                    cells_to_remove.append(i)
                    reward -= 0.5
                    self._create_cell_destruction_particles(cell.pos, cell.radius, cell.color)
                    # sfx: Cell_Pop_Sound
                    break
            if i in cells_to_remove: continue

            # Exit collision
            if cell.pos.distance_to(self.exit_pos) < cell.radius + self.exit_radius:
                self.game_over = True
                reward += 100 # Win condition
                # sfx: Victory_Sound
                break
        
        if cells_to_remove:
            self.cells = [c for i, c in enumerate(self.cells) if i not in set(cells_to_remove)]
            if self.cells:
                self.selected_cell_idx %= len(self.cells)

        # --- Reward Calculation (Distance) ---
        if self.cells:
            avg_pos = self._get_average_cell_position()
            dist = avg_pos.distance_to(self.exit_pos)
            reward += (self.last_distance_to_exit - dist) * 0.01 # Reward for getting closer
            self.last_distance_to_exit = dist
        
        # --- Termination Check ---
        if not self.cells:
            self.game_over = True
            reward -= 100 # Loss condition
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 100 == 0:
            self.obstacle_density = min(0.2, self.obstacle_density + 0.005)
            self._spawn_obstacles(new_only=True)

        self.score += reward
        terminated = self.game_over and not truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_average_cell_position(self):
        if not self.cells:
            return pygame.math.Vector2(self.WORLD_W/2, self.WORLD_H/2)
        return sum([c.pos for c in self.cells], pygame.math.Vector2()) / len(self.cells)

    def _update_camera(self):
        if not self.cells: return
        target_pos = self._get_average_cell_position() - pygame.math.Vector2(self.SCREEN_W / 2, self.SCREEN_H / 2)
        # Smooth camera movement (lerp)
        self.camera_pos = self.camera_pos.lerp(target_pos, 0.1)

    def _world_to_screen(self, world_pos):
        return world_pos - self.camera_pos

    def _spawn_obstacles(self, new_only=False):
        if not new_only:
            self.obstacles = []
        
        num_obstacles = int((self.WORLD_W * self.WORLD_H) / (50*50) * self.obstacle_density)
        
        for _ in range(num_obstacles):
            pos = pygame.math.Vector2(
                self.np_random.uniform(0, self.WORLD_W),
                self.np_random.uniform(0, self.WORLD_H)
            )
            # Avoid spawning on player start or exit
            if pos.distance_to(pygame.math.Vector2(self.WORLD_W/2, self.WORLD_H/2)) < 300 or \
               pos.distance_to(self.exit_pos) < 300:
                continue

            radius = self.np_random.uniform(20, 80)
            self.obstacles.append({'pos': pos, 'radius': radius})

    def _create_cell_destruction_particles(self, pos, radius, color):
        num_particles = int(radius * 1.5)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            p_radius = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(20, 40)
            p_color = (
                max(0, min(255, color[0] + self.np_random.integers(-20, 20))),
                max(0, min(255, color[1] + self.np_random.integers(-20, 20))),
                max(0, min(255, color[2] + self.np_random.integers(-20, 20)))
            )
            self.particles.append(Particle(pos, vel, p_radius, p_color, lifespan))

    def _get_observation(self):
        self._update_camera()
        
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        # Add some faint background details for atmosphere
        for i in range(10):
            # Hash the step count to get pseudo-random but deterministic positions
            x = int(hash(f"bg_x_{i}_{self.steps//100}") % (self.WORLD_W + 400) - 200)
            y = int(hash(f"bg_y_{i}_{self.steps//100}") % (self.WORLD_H + 400) - 200)
            r = int(hash(f"bg_r_{i}_{self.steps//100}") % 150 + 50)
            screen_pos = self._world_to_screen(pygame.math.Vector2(x, y))
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), r, (25, 30, 45))

        # --- Render Game Elements (World Space) ---
        self._render_game()
        
        # --- Render UI (Screen Space) ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            screen_pos = self._world_to_screen(p.pos)
            # Simple alpha blending not supported by default on surfaces without per-pixel alpha
            # We will just draw the circle without alpha
            pygame.draw.circle(self.screen, p.color, (int(screen_pos.x), int(screen_pos.y)), int(p.radius))

        # Render obstacles
        for o in self.obstacles:
            screen_pos = self._world_to_screen(o['pos'])
            if -o['radius'] < screen_pos.x < self.SCREEN_W + o['radius'] and \
               -o['radius'] < screen_pos.y < self.SCREEN_H + o['radius']:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(o['radius']), self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(o['radius']), self.COLOR_OBSTACLE)

        # Render exit
        exit_screen_pos = self. _world_to_screen(self.exit_pos)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 10
        radius = int(self.exit_radius + pulse)
        pygame.gfxdraw.filled_circle(self.screen, int(exit_screen_pos.x), int(exit_screen_pos.y), radius, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, int(exit_screen_pos.x), int(exit_screen_pos.y), radius, (255, 255, 255))
        
        # Render cells
        for i, cell in enumerate(self.cells):
            screen_pos = self._world_to_screen(cell.pos)
            r = int(cell.radius)
            # Main cell body
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), r, cell.color)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), r, tuple(int(c*0.8) for c in cell.color))
            
            # Highlight selected cell
            if i == self.selected_cell_idx:
                glow_r = r + 4 + (math.sin(self.steps * 0.2) + 1)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), int(glow_r), self.COLOR_SELECTED_GLOW)


    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, pos, shadow_color=(0,0,0)):
            text_surface = font.render(text, True, color)
            shadow_surface = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surface, pos)

        # Top-left: Cell count
        draw_text(f"Cells: {len(self.cells)}", self.font_large, self.COLOR_UI_TEXT, (10, 10))
        
        # Top-right: Distance to exit
        dist_str = f"Distance: {int(self.last_distance_to_exit)}"
        draw_text(dist_str, self.font_large, self.COLOR_UI_TEXT, (self.SCREEN_W - self.font_large.size(dist_str)[0] - 10, 10))
        
        # Bottom-left: Score
        draw_text(f"Score: {self.score:.1f}", self.font_small, self.COLOR_UI_TEXT, (10, self.SCREEN_H - 30))

        # Bottom-right: Steps
        draw_text(f"Steps: {self.steps}/{self.MAX_STEPS}", self.font_small, self.COLOR_UI_TEXT, (self.SCREEN_W - 150, self.SCREEN_H - 30))
        
        if self.game_over:
            msg = "VICTORY!" if len(self.cells) > 0 else "GAME OVER"
            color = self.COLOR_EXIT if len(self.cells) > 0 else (255, 50, 50)
            size = self.font_large.size(msg)
            draw_text(msg, self.font_large, color, (self.SCREEN_W/2 - size[0]/2, self.SCREEN_H/2 - size[1]/2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cells_remaining": len(self.cells),
            "distance_to_exit": self.last_distance_to_exit,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    pygame.display.set_caption("Cell Cluster")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # sfx: Reset_Sound
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()