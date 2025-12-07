
# Generated: 2025-08-27T22:04:56.902671
# Source Brief: brief_03006.md
# Brief Index: 3006

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to plant a seed and shift to water a plot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate a thriving grid garden by strategically planting and watering seeds to grow 10 flowers before running out of water."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 5
    UI_PANEL_HEIGHT = 60
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - UI_PANEL_HEIGHT

    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = GAME_AREA_HEIGHT // GRID_ROWS

    # Plant stages
    PLANT_STAGE_EMPTY = 0
    PLANT_STAGE_SEED = 1
    PLANT_STAGE_SPROUT = 2
    PLANT_STAGE_SMALL = 3
    PLANT_STAGE_MEDIUM = 4
    PLANT_STAGE_LARGE = 5
    PLANT_STAGE_FLOWER = 6

    # Game parameters
    INITIAL_WATER = 50
    INITIAL_SEEDS = 10
    WIN_FLOWERS = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_UI_BG = (10, 20, 30)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_WATER_DROP = (100, 150, 255)
    COLOR_SEED = (139, 69, 19)
    PLANT_COLORS = [
        (152, 251, 152),  # Sprout
        (50, 205, 50),    # Small
        (34, 139, 34),    # Medium
        (0, 100, 0),      # Large
    ]
    FLOWER_COLORS = [
        (255, 20, 147), (255, 105, 180), (255, 0, 0), (255, 165, 0),
        (255, 255, 0), (138, 43, 226), (75, 0, 130)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 64)

        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.water_level = self.INITIAL_WATER
        self.seed_count = self.INITIAL_SEEDS
        self.flower_count = 0
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.end_message = ""

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0

        self._update_particles()

        movement = action[0]
        plant_action = action[1] == 1
        water_action = action[2] == 1

        # 1. Handle Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        cx, cy = self.cursor_pos
        plot_state = self.grid[cy, cx]

        # 2. Handle Planting (Space)
        if plant_action and self.seed_count > 0 and plot_state == self.PLANT_STAGE_EMPTY:
            self.grid[cy, cx] = self.PLANT_STAGE_SEED
            self.seed_count -= 1
            # sfx: seed_plant
            self._create_plant_effect(cx, cy)
        
        # 3. Handle Watering (Shift)
        elif water_action and self.water_level > 0 and self.PLANT_STAGE_EMPTY < plot_state < self.PLANT_STAGE_FLOWER:
            self.water_level -= 1
            self.grid[cy, cx] += 1
            reward += 0.1
            # sfx: water_splash
            self._create_water_effect(cx, cy)
            
            if self.grid[cy, cx] == self.PLANT_STAGE_FLOWER:
                self.flower_count += 1
                reward += 1.0
                self.score += 10 # Game score, separate from RL reward
                # sfx: flower_bloom_chime
                self._create_bloom_effect(cx, cy)

        terminated = self._check_termination()
        if terminated:
            if self.flower_count >= self.WIN_FLOWERS:
                reward += 100
                self.score += 1000
                self.end_message = "GARDEN COMPLETE!"
            elif self.water_level <= 0:
                reward -= 100
                self.end_message = "OUT OF WATER!"
            else: # Max steps
                self.end_message = "TIME'S UP!"
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.flower_count >= self.WIN_FLOWERS:
            self.game_over = True
            return True
        if self.water_level <= 0:
            # Check if any plants can still be watered (this check is not strictly needed by brief but is good practice)
            can_grow = np.any((self.grid > self.PLANT_STAGE_EMPTY) & (self.grid < self.PLANT_STAGE_FLOWER))
            if not can_grow:
                 self.game_over = True
                 return True
            # If we are out of water, we lose.
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_plants()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_end_message()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "water": self.water_level,
            "seeds": self.seed_count,
            "flowers": self.flower_count,
        }

    # --- Rendering Helpers ---
    def _render_grid(self):
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.GAME_AREA_HEIGHT), 1)
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)

    def _render_plants(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                stage = self.grid[y, x]
                if stage > self.PLANT_STAGE_EMPTY:
                    self._render_plant_at(x, y, stage)

    def _render_plant_at(self, x, y, stage):
        cx = int(x * self.CELL_WIDTH + self.CELL_WIDTH / 2)
        cy = int(y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)

        if stage == self.PLANT_STAGE_SEED:
            pygame.gfxdraw.filled_circle(self.screen, cx, cy + 15, 5, self.COLOR_SEED)
        
        elif self.PLANT_STAGE_SEED < stage < self.PLANT_STAGE_FLOWER:
            plant_idx = stage - self.PLANT_STAGE_SPROUT
            color = self.PLANT_COLORS[plant_idx]
            
            # Stem
            stem_height = 10 + plant_idx * 5
            pygame.draw.line(self.screen, color, (cx, cy + 15), (cx, cy + 15 - stem_height), 3)

            # Leaves
            for i in range(plant_idx + 1):
                leaf_size = 3 + i
                leaf_y = cy + 10 - i * 4
                pygame.gfxdraw.filled_circle(self.screen, cx - leaf_size, leaf_y, leaf_size, color)
                pygame.gfxdraw.filled_circle(self.screen, cx + leaf_size, leaf_y, leaf_size, color)

        elif stage == self.PLANT_STAGE_FLOWER:
            # Stem
            pygame.draw.line(self.screen, self.PLANT_COLORS[-1], (cx, cy + 15), (cx, cy - 10), 3)
            # Flower
            flower_color_idx = (x + y * self.GRID_COLS) % len(self.FLOWER_COLORS)
            color = self.FLOWER_COLORS[flower_color_idx]
            
            # Petals
            num_petals = 6
            for i in range(num_petals):
                angle = 2 * math.pi * i / num_petals
                px = int(cx + math.cos(angle) * 12)
                py = int(cy - 10 + math.sin(angle) * 12)
                pygame.gfxdraw.filled_circle(self.screen, px, py, 8, color)
            
            # Center
            pygame.gfxdraw.filled_circle(self.screen, cx, cy - 10, 6, (255, 223, 0))

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        
        # Pulsing alpha effect
        alpha = 100 + 60 * math.sin(self.steps * 0.3)
        color = (*self.COLOR_CURSOR, alpha)

        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(s, color, s.get_rect(), border_radius=8)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), 3, border_radius=8)
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        ui_rect = pygame.Rect(0, self.GAME_AREA_HEIGHT, self.SCREEN_WIDTH, self.UI_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.GAME_AREA_HEIGHT), (self.SCREEN_WIDTH, self.GAME_AREA_HEIGHT), 2)

        y_pos = self.GAME_AREA_HEIGHT + self.UI_PANEL_HEIGHT / 2

        # Water
        water_text = self.font_medium.render(f"💧 {self.water_level}", True, self.COLOR_TEXT)
        self.screen.blit(water_text, (30, y_pos - water_text.get_height() / 2))

        # Seeds
        seed_text = self.font_medium.render(f"🌰 {self.seed_count}", True, self.COLOR_TEXT)
        self.screen.blit(seed_text, (180, y_pos - seed_text.get_height() / 2))

        # Flowers
        flower_text = self.font_medium.render(f"🌸 {self.flower_count} / {self.WIN_FLOWERS}", True, self.COLOR_TEXT)
        self.screen.blit(flower_text, (330, y_pos - flower_text.get_height() / 2))

        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 30 - score_text.get_width(), y_pos - score_text.get_height() / 2))

    def _render_end_message(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        
        text_surf = self.font_large.render(self.end_message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        s.blit(text_surf, text_rect)
        self.screen.blit(s, (0, 0))

    # --- Particle System ---
    def _create_water_effect(self, grid_x, grid_y):
        cx = grid_x * self.CELL_WIDTH + self.CELL_WIDTH / 2
        cy = grid_y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - 1] # Move up slightly
            lifespan = random.randint(15, 25)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': lifespan, 'color': self.COLOR_WATER_DROP, 'type': 'drop'})

    def _create_plant_effect(self, grid_x, grid_y):
        cx = grid_x * self.CELL_WIDTH + self.CELL_WIDTH / 2
        cy = grid_y * self.CELL_HEIGHT + self.CELL_HEIGHT
        for _ in range(5):
            vel = [random.uniform(-1, 1), random.uniform(-2, -0.5)]
            lifespan = random.randint(10, 20)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': lifespan, 'color': self.COLOR_SEED, 'type': 'spark'})
            
    def _create_bloom_effect(self, grid_x, grid_y):
        cx = grid_x * self.CELL_WIDTH + self.CELL_WIDTH / 2
        cy = grid_y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2 - 10
        color_idx = (grid_x + grid_y * self.GRID_COLS) % len(self.FLOWER_COLORS)
        color = self.FLOWER_COLORS[color_idx]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 30)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': lifespan, 'color': color, 'type': 'spark'})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if p['type'] == 'drop':
                p['vel'][1] += 0.1 # Gravity for water
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p['life'] * 10)
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))

            if p['type'] == 'drop':
                pygame.draw.circle(self.screen, p['color'], pos, max(0, p['life'] // 5))
            elif p['type'] == 'spark':
                size = max(0, p['life'] // 4)
                rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
                pygame.draw.rect(self.screen, p['color'], rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Garden Grid")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement = 0 # no-op
        plant = 0
        water = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: plant = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: water = 1

        action = [movement, plant, water]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            env.reset()

        # Since auto_advance is False, we control the step rate
        clock.tick(10) # Human play speed

    env.close()