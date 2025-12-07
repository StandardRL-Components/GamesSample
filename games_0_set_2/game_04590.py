
# Generated: 2025-08-28T02:51:53.580061
# Source Brief: brief_04590.md
# Brief Index: 4590

        
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
        "Controls: Use arrow keys to move the selector. Press Space to plant a seed on an empty plot, or water an existing plant."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate a garden by planting seeds and watering them to maturity. Manage your limited water supply to grow 10 mature plants to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_COLS = 10
        self.GRID_ROWS = 6
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.screen_width - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.screen_height - self.GRID_HEIGHT) // 2 + 20

        # Plant stages
        self.EMPTY = 0
        self.SEED = 1
        self.SPROUT = 2
        self.SMALL_PLANT = 3
        self.MATURE = 4
        self.GROWTH_STAGES = 4 # Number of stages from seed to mature

        self.INITIAL_WATER = 35 # Sufficient for 10 plants (3 waterings each) + a small buffer
        self.WIN_CONDITION = 10
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 25)
        self.COLOR_GRID_BG = (87, 58, 46)
        self.COLOR_GRID_LINES = (66, 42, 31)
        self.COLOR_WATER = (50, 150, 255)
        self.COLOR_UI_BG = (10, 15, 12)
        self.COLOR_UI_FG = (220, 220, 220)
        self.COLOR_UI_SHADOW = (0, 0, 0)
        self.COLOR_PLANT_STEM = (50, 180, 50)
        self.COLOR_PLANT_LEAF = (80, 220, 80)
        self.COLOR_FLOWER = (255, 100, 100)
        self.COLOR_FLOWER_CENTER = (255, 255, 0)
        self.COLOR_SEED = (139, 69, 19)

        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.grid = None
        self.selector_pos = None
        self.water = 0
        self.mature_plants = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.selector_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.water = self.INITIAL_WATER
        self.mature_plants = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        reward = 0.0

        # 1. Handle Movement
        if movement == 1:  # Up
            self.selector_pos[1] = (self.selector_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2:  # Down
            self.selector_pos[1] = (self.selector_pos[1] + 1) % self.GRID_ROWS
        elif movement == 3:  # Left
            self.selector_pos[0] = (self.selector_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4:  # Right
            self.selector_pos[0] = (self.selector_pos[0] + 1) % self.GRID_COLS

        # 2. Handle Action (Plant/Water)
        if space_pressed:
            sel_x, sel_y = self.selector_pos
            plot_state = self.grid[sel_y, sel_x]

            if plot_state == self.EMPTY:
                # Plant a seed
                self.grid[sel_y, sel_x] = self.SEED
                # SFX: plant_seed.wav
            elif plot_state < self.MATURE:
                # Water a plant
                if self.water > 0:
                    self.water -= 1
                    self.grid[sel_y, sel_x] += 1
                    reward += 0.1
                    self._add_water_particles(sel_x, sel_y)
                    # SFX: water_splash.wav
                    if self.grid[sel_y, sel_x] == self.MATURE:
                        self.mature_plants += 1
                        reward += 1.0
                        self.score += 1
                        # SFX: plant_mature.wav

        # 3. Update game logic
        self._update_particles()
        self.steps += 1
        
        # 4. Check for termination
        terminated = False
        if self.mature_plants >= self.WIN_CONDITION:
            terminated = True
            reward += 10.0
            self.game_over = True
            self.win_message = "VICTORY!"
        elif self.water <= 0 and self.mature_plants < self.WIN_CONDITION:
            # Check if any plant can still be watered (this condition is met if water is 0)
            terminated = True
            reward -= 10.0
            self.game_over = True
            self.win_message = "WATER DEPLETED"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _add_water_particles(self, grid_x, grid_y):
        cx = self.GRID_X_OFFSET + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        cy = self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            particle = {
                "pos": [cx, cy],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                "life": 1.0,
                "size": self.np_random.uniform(2, 5)
            }
            self.particles.append(particle)
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 0.03
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_garden()
        self._render_selector()
        self._render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_garden(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH, self.GRID_HEIGHT))

        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                rect = (self.GRID_X_OFFSET + x * self.CELL_SIZE,
                        self.GRID_Y_OFFSET + y * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

                # Draw plant
                state = self.grid[y, x]
                center_x = rect[0] + self.CELL_SIZE // 2
                center_y = rect[1] + self.CELL_SIZE // 2

                if state == self.SEED:
                    pygame.draw.circle(self.screen, self.COLOR_SEED, (center_x, center_y + 10), 3)
                elif state == self.SPROUT:
                    pygame.draw.aaline(self.screen, self.COLOR_PLANT_STEM, (center_x, center_y + 10), (center_x, center_y))
                    pygame.draw.aaline(self.screen, self.COLOR_PLANT_LEAF, (center_x, center_y), (center_x - 5, center_y - 5))
                    pygame.draw.aaline(self.screen, self.COLOR_PLANT_LEAF, (center_x, center_y), (center_x + 5, center_y - 5))
                elif state == self.SMALL_PLANT:
                    # Stem
                    pygame.draw.line(self.screen, self.COLOR_PLANT_STEM, (center_x, center_y + 15), (center_x, center_y - 5), 3)
                    # Leaves
                    pygame.draw.ellipse(self.screen, self.COLOR_PLANT_LEAF, (center_x - 12, center_y, 12, 8))
                    pygame.draw.ellipse(self.screen, self.COLOR_PLANT_LEAF, (center_x, center_y + 5, 12, 8))
                elif state == self.MATURE:
                    # Stem
                    pygame.draw.line(self.screen, self.COLOR_PLANT_STEM, (center_x, center_y + 15), (center_x, center_y - 10), 3)
                    # Flower
                    flower_y = center_y - 12
                    for i in range(5):
                        angle = 2 * math.pi * i / 5 + (self.steps * 0.05)
                        px = center_x + math.cos(angle) * 10
                        py = flower_y + math.sin(angle) * 10
                        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 7, self.COLOR_FLOWER)
                        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), 7, self.COLOR_FLOWER)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, flower_y, 5, self.COLOR_FLOWER_CENTER)
                    pygame.gfxdraw.aacircle(self.screen, center_x, flower_y, 5, self.COLOR_FLOWER_CENTER)

    def _render_selector(self):
        sel_x, sel_y = self.selector_pos
        rect = (self.GRID_X_OFFSET + sel_x * self.CELL_SIZE,
                self.GRID_Y_OFFSET + sel_y * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing glow effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        color = (255, 255, 100 + 120 * pulse)
        
        # Draw a thick, bright border
        pygame.draw.rect(self.screen, color, rect, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 255)))
            color = (*self.COLOR_WATER, alpha)
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

    def _render_ui(self):
        # Water bar
        water_bar_width = 200
        water_bar_height = 20
        water_bar_x = (self.screen_width - water_bar_width) // 2
        water_bar_y = 20
        
        water_ratio = max(0, self.water / self.INITIAL_WATER)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (water_bar_x, water_bar_y, water_bar_width, water_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_WATER, (water_bar_x, water_bar_y, water_bar_width * water_ratio, water_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_FG, (water_bar_x, water_bar_y, water_bar_width, water_bar_height), 1)
        
        water_text = f"WATER: {self.water}/{self.INITIAL_WATER}"
        self._draw_text(water_text, (water_bar_x + water_bar_width // 2, water_bar_y - 15), self.font_ui, self.COLOR_UI_FG, self.COLOR_UI_SHADOW)

        # Mature plants score
        score_text = f"GROWN: {self.mature_plants}/{self.WIN_CONDITION}"
        self._draw_text(score_text, (self.screen_width - 100, 25), self.font_ui, self.COLOR_UI_FG, self.COLOR_UI_SHADOW)
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        self._draw_text(self.win_message, (self.screen_width // 2, self.screen_height // 2), self.font_big, self.COLOR_UI_FG, self.COLOR_UI_SHADOW)

    def _draw_text(self, text, pos, font, color, shadow_color=None):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=pos)
        if shadow_color:
            shadow_surf = font.render(text, True, shadow_color)
            shadow_rect = shadow_surf.get_rect(center=(pos[0] + 2, pos[1] + 2))
            self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "water": self.water,
            "mature_plants": self.mature_plants
        }

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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Grid Garden")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Game loop for human play
    while not terminated:
        movement = 0 # No-op
        space_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit
                     terminated = True

        # For turn-based, we only step if an action was taken
        if movement != 0 or space_pressed != 0:
            action = [movement, space_pressed, 0] # Shift is not used
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()