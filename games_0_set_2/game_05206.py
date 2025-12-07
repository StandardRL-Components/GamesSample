
# Generated: 2025-08-28T04:17:50.214957
# Source Brief: brief_05206.md
# Brief Index: 5206

        
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
        "Controls: Arrow keys to move cursor. Space to plant a seed. Shift to water adjacent tiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate a garden by planting seeds and watering them to grow flowers. Manage your limited water supply to reach the goal of 5 flowers!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 10
    GRID_ROWS = 6
    CELL_SIZE = 50
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # Colors
    COLOR_BG = (87, 58, 39)          # Dark soil
    COLOR_GRID = (110, 78, 59)       # Lighter soil for grid lines
    COLOR_CURSOR = (255, 220, 0)     # Bright yellow
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_SEED = (139, 69, 19)       # Brown
    COLOR_SPROUT = (102, 204, 0)     # Bright green
    COLOR_WATER_DROP = (59, 142, 234)
    FLOWER_COLORS = [
        (255, 51, 51),   # Red
        (255, 255, 102), # Yellow
        (204, 102, 255), # Purple
        (255, 153, 204)  # Pink
    ]
    
    # Game parameters
    INITIAL_WATER = 20
    INITIAL_SEEDS = 10
    WIN_CONDITION_FLOWERS = 5
    MAX_STEPS = 500
    
    # Plant stages
    PLANT_EMPTY = 0
    PLANT_SEED = 1
    PLANT_SPROUT = 2
    PLANT_SMALL_PLANT = 3
    PLANT_FLOWER = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.water = 0
        self.seeds = 0
        self.flowers_grown = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.particles = []
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.grid = [[{'state': self.PLANT_EMPTY, 'color': None} for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.water = self.INITIAL_WATER
        self.seeds = self.INITIAL_SEEDS
        self.flowers_grown = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Update particles from previous frame
        self._update_particles()
        
        # --- Action Handling (Priority: Plant > Water > Move) ---
        action_taken = False
        if space_held: # Plant
            x, y = self.cursor_pos
            if self.seeds > 0 and self.grid[y][x]['state'] == self.PLANT_EMPTY:
                self.seeds -= 1
                self.grid[y][x]['state'] = self.PLANT_SEED
                # Sound: Plant seed
                action_taken = True
        
        elif shift_held: # Water
            if self.water > 0:
                self.water -= 1
                # Sound: Water splash
                self._spawn_water_particles()
                
                cursor_x, cursor_y = self.cursor_pos
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cursor_x + dx, cursor_y + dy
                    if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                        cell = self.grid[ny][nx]
                        if self.PLANT_SEED <= cell['state'] < self.PLANT_FLOWER:
                            cell['state'] += 1
                            reward += 0.1 # Reward for advancing growth
                            if cell['state'] == self.PLANT_FLOWER:
                                self.flowers_grown += 1
                                reward += 1.0 # Reward for growing a flower
                                color_idx = self.np_random.integers(0, len(self.FLOWER_COLORS))
                                cell['color'] = self.FLOWER_COLORS[color_idx]
                                # Sound: Flower bloom
                action_taken = True

        if not action_taken: # Move
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        self.score += reward
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += term_reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.flowers_grown >= self.WIN_CONDITION_FLOWERS:
            self.game_over = True
            self.game_over_message = "GARDEN COMPLETE!"
            return True, 10.0 # Victory reward
        
        # Check if any further growth is possible
        can_grow = False
        if self.water > 0:
            for y in range(self.GRID_ROWS):
                for x in range(self.GRID_COLS):
                    if self.PLANT_SEED <= self.grid[y][x]['state'] < self.PLANT_FLOWER:
                        can_grow = True
                        break
                if can_grow:
                    break
        
        if not can_grow and self.water <= 0:
            self.game_over = True
            self.game_over_message = "OUT OF WATER"
            return True, -10.0 # Failure penalty

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_over_message = "TIME'S UP"
            return True, 0

        return False, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "water": self.water, "seeds": self.seeds, "flowers": self.flowers_grown}

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), 2)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), 2)

        # Draw plants
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                cell = self.grid[r][c]
                if cell['state'] != self.PLANT_EMPTY:
                    self._draw_plant(c, r, cell['state'], cell['color'])

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_WATER_DROP, (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

        # Draw cursor
        cursor_x = self.GRID_X + self.cursor_pos[0] * self.CELL_SIZE
        cursor_y = self.GRID_Y + self.cursor_pos[1] * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE), 4)

    def _draw_plant(self, c, r, state, color):
        center_x = self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        
        if state == self.PLANT_SEED:
            pygame.draw.circle(self.screen, self.COLOR_SEED, (center_x, center_y + 15), 5)
        
        elif state == self.PLANT_SPROUT:
            pygame.draw.line(self.screen, self.COLOR_SPROUT, (center_x, center_y + 20), (center_x, center_y + 5), 3)
            pygame.draw.circle(self.screen, self.COLOR_SPROUT, (center_x - 5, center_y + 5), 4)
            pygame.draw.circle(self.screen, self.COLOR_SPROUT, (center_x + 5, center_y + 5), 4)

        elif state == self.PLANT_SMALL_PLANT:
            pygame.draw.line(self.screen, self.COLOR_SPROUT, (center_x, center_y + 20), (center_x, center_y - 5), 4)
            pygame.gfxdraw.filled_ellipse(self.screen, center_x - 10, center_y, 8, 4, self.COLOR_SPROUT)
            pygame.gfxdraw.filled_ellipse(self.screen, center_x + 10, center_y, 8, 4, self.COLOR_SPROUT)
            pygame.gfxdraw.filled_ellipse(self.screen, center_x - 8, center_y + 10, 8, 4, self.COLOR_SPROUT)
            pygame.gfxdraw.filled_ellipse(self.screen, center_x + 8, center_y + 10, 8, 4, self.COLOR_SPROUT)

        elif state == self.PLANT_FLOWER:
            # Stem
            pygame.draw.line(self.screen, self.COLOR_SPROUT, (center_x, center_y + 20), (center_x, center_y - 10), 4)
            # Leaves
            pygame.gfxdraw.filled_ellipse(self.screen, center_x - 12, center_y + 10, 10, 5, self.COLOR_SPROUT)
            pygame.gfxdraw.filled_ellipse(self.screen, center_x + 12, center_y + 10, 10, 5, self.COLOR_SPROUT)
            # Petals
            for i in range(5):
                angle = (i / 5) * 2 * math.pi
                px = center_x + int(math.cos(angle) * 10)
                py = center_y - 10 + int(math.sin(angle) * 10)
                pygame.gfxdraw.filled_circle(self.screen, px, py, 8, color)
                pygame.gfxdraw.aacircle(self.screen, px, py, 8, (0,0,0,50))
            # Center of flower
            pygame.draw.circle(self.screen, (255, 200, 0), (center_x, center_y - 10), 6)

    def _render_ui(self):
        # --- UI Backgrounds for readability ---
        s = pygame.Surface((160, 40))
        s.set_alpha(128)
        s.fill((0,0,0))
        self.screen.blit(s, (10,10))
        self.screen.blit(s, (self.SCREEN_WIDTH - 170, 10))
        self.screen.blit(s, (10, self.SCREEN_HEIGHT - 50))
        
        # --- Water UI ---
        water_text = self.font_ui.render(f"Water: {self.water}", True, self.COLOR_UI_TEXT)
        self.screen.blit(water_text, (20, 20))
        
        # --- Flowers UI ---
        flower_text = self.font_ui.render(f"Flowers: {self.flowers_grown} / {self.WIN_CONDITION_FLOWERS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(flower_text, (self.SCREEN_WIDTH - 160, 20))

        # --- Seeds UI ---
        seed_text = self.font_ui.render(f"Seeds: {self.seeds}", True, self.COLOR_UI_TEXT)
        self.screen.blit(seed_text, (20, self.SCREEN_HEIGHT - 40))

        # --- Game Over Message ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            s.set_alpha(180)
            s.fill((0,0,0))
            self.screen.blit(s, (0,0))
            
            message_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            text_rect = message_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(message_surf, text_rect)

    def _spawn_water_particles(self):
        cx = self.GRID_X + self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        cy = self.GRID_Y + self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            start_pos = [cx + math.cos(angle) * 20, cy + math.sin(angle) * 20]
            self.particles.append({
                'pos': start_pos,
                'vel': [random.uniform(-1, 1), random.uniform(1, 3)], # Gravity effect
                'life': random.randint(15, 25),
                'size': random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95 # Shrink
        self.particles = [p for p in self.particles if p['life'] > 0]

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*40)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*40 + "\n")

    # Use a separate display for human play
    pygame.display.set_caption("Garden Simulator")
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
        
        # Buttons
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Info: {info}")

        # --- Render to human display ---
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need a small delay for human playability
        pygame.time.wait(100)

    print("Game Over!")
    pygame.quit()