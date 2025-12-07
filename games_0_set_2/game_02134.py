
# Generated: 2025-08-27T19:22:33.443086
# Source Brief: brief_02134.md
# Brief Index: 2134

        
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
        "Controls: Arrow keys to move cursor. Space to plant, water, or harvest."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically plant and water seeds in a grid-based garden to harvest a bountiful crop before running out of water."
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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.grid_rows, self.grid_cols = 5, 8
        self.max_water = 100
        self.harvest_goal = 20
        self.max_steps = 1000
        
        # Plant states: 0:empty, 1:seed, 2:sprout, 3:growing, 4:mature
        self.plant_state_max = 4

        # Visuals
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 72)
        
        self.color_bg = (50, 30, 20)
        self.color_soil = (92, 64, 51)
        self.color_grid = (70, 50, 40)
        self.color_cursor = (255, 255, 0)
        self.color_water_bar = (0, 150, 255)
        self.color_water_drop = (100, 180, 255)
        self.color_text = (240, 240, 240)
        self.color_harvest = (255, 200, 0)
        
        self.plant_colors = [
            self.color_soil,         # 0: Empty
            (100, 200, 100),       # 1: Seed
            (80, 220, 80),         # 2: Sprout
            (60, 240, 60),         # 3: Growing
            self.color_harvest,    # 4: Mature
        ]

        # Grid layout calculation
        self.cell_size = 48
        self.grid_width = self.grid_cols * self.cell_size
        self.grid_height = self.grid_rows * self.cell_size
        self.grid_offset_x = (self.screen_width - self.grid_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_height) // 2 + 30

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.water_level = None
        self.harvested_count = None
        self.particles = None
        self.message = None
        self.steps = None
        self.score = None
        self.game_over = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.message = ""

        self.grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int8)
        self.cursor_pos = [self.grid_rows // 2, self.grid_cols // 2]
        self.water_level = self.max_water
        self.harvested_count = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        
        reward = 0
        
        # Handle cursor movement
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.grid_rows
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.grid_rows
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.grid_cols
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.grid_cols

        # Handle main action (plant/water/harvest)
        if space_pressed:
            reward += self._perform_grid_action()

        self._update_particles()
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.harvested_count >= self.harvest_goal:
                reward += 100
                self.message = "YOU WIN!"
            elif self.water_level <= 0:
                reward -= 100
                self.message = "NO WATER!"
            else: # Max steps
                self.message = "TIME UP!"
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _perform_grid_action(self):
        if self.water_level <= 0:
            return 0 # Can't do anything without water

        self.water_level -= 1
        # sfx: action_sound
        
        row, col = self.cursor_pos
        cell_state = self.grid[row, col]
        reward = 0

        if cell_state == 0: # Empty cell -> Plant
            self.grid[row, col] = 1
            # sfx: plant_seed
            self._create_particles(self.cursor_pos, 'plant')
            reward -= 0.1 # Penalty for action on empty cell
        elif cell_state < self.plant_state_max: # Growing plant -> Water
            self.grid[row, col] += 1
            # sfx: water_plant
            self._create_particles(self.cursor_pos, 'water')
            reward += 1
        elif cell_state == self.plant_state_max: # Mature plant -> Harvest
            self.grid[row, col] = 0
            self.harvested_count += 1
            self.score += 10
            reward += 10
            # sfx: harvest_plant
            self._create_particles(self.cursor_pos, 'harvest')
        
        return reward

    def _check_termination(self):
        return (
            self.harvested_count >= self.harvest_goal or
            self.water_level <= 0 or
            self.steps >= self.max_steps
        )

    def _get_observation(self):
        self.screen.fill(self.color_bg)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and plants
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell_rect = pygame.Rect(
                    self.grid_offset_x + c * self.cell_size,
                    self.grid_offset_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.screen, self.color_soil, cell_rect)
                pygame.draw.rect(self.screen, self.color_grid, cell_rect, 1)

                state = self.grid[r, c]
                if state > 0:
                    center_x = cell_rect.centerx
                    center_y = cell_rect.centery
                    radius = 5 + state * 4
                    color = self.plant_colors[state]
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.cell_size,
            self.grid_offset_y + cursor_r * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.color_cursor, cursor_rect, 3)

        self._render_particles()
    
    def _render_ui(self):
        # Water Bar
        bar_width = 200
        bar_height = 20
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = 20
        water_ratio = max(0, self.water_level / self.max_water)
        fill_width = int(bar_width * water_ratio)
        
        pygame.draw.rect(self.screen, self.color_grid, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.color_water_bar, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.color_text, (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Text info
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.color_text)
        harvest_text = self.font_ui.render(f"Harvested: {self.harvested_count}/{self.harvest_goal}", True, self.color_text)
        
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(harvest_text, (self.screen_width - harvest_text.get_width() - 20, 20))

        # Game Over Message
        if self.game_over and self.message:
            msg_surf = self.font_msg.render(self.message, True, self.color_text)
            msg_rect = msg_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, grid_pos, p_type):
        r, c = grid_pos
        center_x = self.grid_offset_x + int((c + 0.5) * self.cell_size)
        center_y = self.grid_offset_y + int((r + 0.5) * self.cell_size)

        if p_type == 'water':
            for _ in range(15):
                angle = random.uniform(0, math.pi * 2)
                speed = random.uniform(0.5, 1.5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
                life = random.randint(15, 25)
                self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'type': 'water'})
        elif p_type == 'harvest':
            for _ in range(30):
                angle = random.uniform(0, math.pi * 2)
                speed = random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = random.randint(20, 35)
                self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'type': 'harvest'})
        elif p_type == 'plant':
            for _ in range(10):
                angle = random.uniform(0, math.pi * 2)
                speed = random.uniform(0.2, 0.8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = random.randint(10, 20)
                self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'type': 'plant'})


    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if p['type'] == 'water':
                p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            life_ratio = p['life'] / 35.0
            
            if p['type'] == 'water':
                size = max(1, int(3 * life_ratio))
                pygame.draw.circle(self.screen, self.color_water_drop, pos, size)
            elif p['type'] == 'harvest':
                size = max(1, int(6 * life_ratio))
                color = self.color_harvest
                pygame.draw.circle(self.screen, color, pos, size)
            elif p['type'] == 'plant':
                size = max(1, int(4 * life_ratio))
                color = self.plant_colors[1]
                pygame.draw.circle(self.screen, color, pos, size)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "water": self.water_level,
            "harvested": self.harvested_count
        }

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Garden Master")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*30)
    print("      Garden Master Test")
    print("="*30)
    print(env.user_guide)
    print("Press ESC or close window to quit.")
    print("="*30 + "\n")
    
    running = True
    while running:
        # Reset action at the start of each frame
        action.fill(0)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Harvested: {info['harvested']}, Water: {info['water']}")

        if done:
            print(f"Game Over! Final Score: {info['score']}")
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Run at 10 FPS for human playability

    pygame.quit()