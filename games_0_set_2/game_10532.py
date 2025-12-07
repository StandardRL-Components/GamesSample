import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:33:08.509833
# Source Brief: brief_00532.md
# Brief Index: 532
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, color, lifespan_range=(20, 40), speed_range=(2, 5), radius_range=(2, 5)):
        self.pos = list(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = random.uniform(*lifespan_range)
        self.max_lifespan = lifespan_range[1]
        self.color = color
        self.radius = random.uniform(*radius_range)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        self.radius *= 0.95

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

class Seed:
    """Represents a plantable seed on the grid."""
    def __init__(self, grid_pos, seed_info, cell_size, grid_offset):
        self.grid_pos = grid_pos
        self.type_info = seed_info
        self.growth = 0.0  # Range 0.0 to 1.0+
        self.is_harvested = False
        self.cell_size = cell_size
        self.grid_offset = grid_offset
        self.pulse_phase = random.uniform(0, math.pi * 2)

    def update(self, dt):
        if self.growth < 1.0:
            self.growth += dt / self.type_info['growth_time']

    def get_screen_pos(self):
        x = self.grid_offset[0] + (self.grid_pos[0] + 0.5) * self.cell_size
        y = self.grid_offset[1] + (self.grid_pos[1] + 0.5) * self.cell_size
        return (x, y)

    def draw(self, surface, total_steps):
        pos = self.get_screen_pos()
        
        if self.growth >= 1.0:
            # Ready to harvest state: yellow and pulsing
            color = (255, 220, 0)
            pulse = 1 + 0.1 * math.sin(total_steps * 0.25 + self.pulse_phase)
            radius = int(self.cell_size * 0.45 * pulse)
            
            # Draw a soft glow effect
            glow_radius = int(radius * 1.6)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, color + (50,), (glow_radius, glow_radius), glow_radius)
            surface.blit(glow_surf, (int(pos[0] - glow_radius), int(pos[1] - glow_radius)))
        else:
            # Growing state: color by type, size by growth
            color = self.type_info['color']
            radius = int(self.cell_size * 0.45 * self.growth)
        
        # Draw the main seed circle with anti-aliasing
        if radius > 0:
            pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), radius, color)
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), radius, color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Plant seeds and create chain reactions to clear the grid before time runs out. "
        "Match seed types for bigger combos and higher scores."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to plant a seed and shift to cycle through seed types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        
        self.GRID_SIZE = 10
        self.GRID_AREA_SIZE = 360
        self.CELL_SIZE = self.GRID_AREA_SIZE // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_AREA_SIZE) // 2 + 10

        # Visual Style
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HARVEST = (255, 220, 0)

        self.SEED_TYPES = {
            0: {'name': 'FAST', 'growth_time': 1.0, 'color': (255, 80, 80)},
            1: {'name': 'MEDIUM', 'growth_time': 2.0, 'color': (80, 120, 255)},
            2: {'name': 'SLOW', 'growth_time': 3.0, 'color': (80, 255, 120)},
        }
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_seed_select = pygame.font.SysFont('Consolas', 16, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.seeds = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_seed_type_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.won = False
        
        # self.reset() is called by the gym wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        self.timer = self.GAME_DURATION_SECONDS
        
        self.seeds = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_seed_type_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        reward += self._handle_input(action)
        reward += self._update_game_state()

        terminated = self.timer <= 0 or self.won
        truncated = False # No step limit other than time
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos = [self.cursor_pos[0] % self.GRID_SIZE, self.cursor_pos[1] % self.GRID_SIZE]

        # Cycle seed type (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_seed_type_idx = (self.selected_seed_type_idx + 1) % len(self.SEED_TYPES)
            # sfx: UI_CYCLE

        # Plant seed (on press)
        if space_held and not self.prev_space_held:
            is_occupied = any(s.grid_pos == self.cursor_pos for s in self.seeds)
            if not is_occupied:
                seed_info = self.SEED_TYPES[self.selected_seed_type_idx]
                new_seed = Seed(list(self.cursor_pos), seed_info, self.CELL_SIZE, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))
                self.seeds.append(new_seed)
                reward += 0.1
                # sfx: PLANT_SEED

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return reward

    def _update_game_state(self):
        dt = 1.0 / self.FPS
        reward = 0

        self.timer = max(0, self.timer - dt)

        for seed in self.seeds:
            seed.update(dt)

        # Harvest Logic
        completed_seeds = [s for s in self.seeds if s.growth >= 1.0 and not s.is_harvested]
        if completed_seeds:
            harvest_queue = deque(completed_seeds)
            harvested_set = set(completed_seeds)
            for s in completed_seeds: s.is_harvested = True

            while harvest_queue:
                current_seed = harvest_queue.popleft()
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor_pos = [current_seed.grid_pos[0] + dx, current_seed.grid_pos[1] + dy]
                    for neighbor_seed in self.seeds:
                        if neighbor_seed.grid_pos == neighbor_pos and not neighbor_seed.is_harvested:
                            if neighbor_seed.type_info == current_seed.type_info:
                                neighbor_seed.growth = 1.0
                                neighbor_seed.is_harvested = True
                                harvested_set.add(neighbor_seed)
                                harvest_queue.append(neighbor_seed)
                                # sfx: CHAIN_REACTION_POP

            num_harvested = len(harvested_set)
            if num_harvested > 0:
                # sfx: HARVEST_SUCCESS
                reward += num_harvested * 1.0
                self.score += num_harvested
                for harvested_seed in harvested_set:
                    pos = harvested_seed.get_screen_pos()
                    for _ in range(20): self.particles.append(Particle(pos, self.COLOR_HARVEST))
                self.seeds = [s for s in self.seeds if not s.is_harvested]

                if not self.seeds:
                    self.won = True
                    reward += 50.0
                    # sfx: LEVEL_CLEAR_FANFARE
        
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles: p.update()
            
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y), (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_AREA_SIZE), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE), (self.GRID_OFFSET_X + self.GRID_AREA_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE), 1)

        for seed in self.seeds:
            seed.draw(self.screen, self.steps)

        # Draw cursor
        cursor_rect = pygame.Rect(self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
        
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        timer_text = self.font_ui.render(f"TIME: {int(self.timer):02}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)
        
        # Selected Seed UI
        selected_info = self.SEED_TYPES[self.selected_seed_type_idx]
        select_text = self.font_seed_select.render("SELECTED SEED", True, self.COLOR_UI_TEXT)
        select_rect = select_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 35))
        self.screen.blit(select_text, select_rect)
        
        seed_name_text = self.font_seed_select.render(f"{selected_info['name']}", True, selected_info['color'])
        seed_name_rect = seed_name_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 15))
        self.screen.blit(seed_name_text, seed_name_rect)

        # Visual indicator for selected seed
        indicator_pos = (seed_name_rect.left - 20, self.SCREEN_HEIGHT - 15)
        pygame.gfxdraw.aacircle(self.screen, int(indicator_pos[0]), int(indicator_pos[1]), 8, selected_info['color'])
        pygame.gfxdraw.filled_circle(self.screen, int(indicator_pos[0]), int(indicator_pos[1]), 8, selected_info['color'])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "seeds_on_grid": len(self.seeds),
            "won": self.won,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and debugging.
    # It will not be executed by the test suite.
    # We can re-enable the display for this.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Seed Sync")
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0

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
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Seeds: {info['seeds_on_grid']}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}" + (" YOU WON!" if info['won'] else ""))
            obs, info = env.reset()
            pygame.time.wait(2000)

        env.clock.tick(env.FPS)

    env.close()