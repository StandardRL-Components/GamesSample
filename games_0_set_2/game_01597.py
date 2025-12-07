
# Generated: 2025-08-28T02:07:14.162054
# Source Brief: brief_01597.md
# Brief Index: 1597

        
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
        "Controls: Arrow keys to move cursor. Space to place a firebreak. "
        "Each action is one turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Contain a raging forest fire for 30 turns by strategically deploying firebreaks. "
        "The fire spreads to adjacent trees each turn. You lose if the fire reaches the edge."
    )

    # Should frames auto-advance or wait for user input?
    # This is a turn-based game, so state advances only on action.
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_TURNS = 30
    
    # Tile states
    TILE_UNBURNT = 0
    TILE_FIRE = 1
    TILE_FIREBREAK = 2
    
    # Colors (Bright and high contrast)
    COLOR_BG = (20, 25, 30) # Dark blue-grey
    COLOR_UNBURNT = (34, 139, 34) # Forest Green
    COLOR_FIRE_1 = (255, 69, 0) # OrangeRed
    COLOR_FIRE_2 = (255, 140, 0) # DarkOrange
    COLOR_FIRE_3 = (255, 215, 0) # Gold
    COLOR_FIREBREAK = (139, 69, 19) # SaddleBrown
    COLOR_CURSOR = (255, 255, 0, 200) # Yellow, semi-transparent
    COLOR_GRID_LINE = (40, 50, 60)
    COLOR_TEXT = (240, 240, 240)
    COLOR_WIN = (60, 220, 60)
    COLOR_LOSE = (220, 60, 60)

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
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        
        # Etc...
        self.grid = None
        self.cursor_pos = None
        self.turns_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None
        
        # Calculate rendering geometry once
        self.grid_render_size = self.SCREEN_HEIGHT - 80 # 320x320
        self.cell_size = self.grid_render_size / self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_size) // 2
        
        # Initialize state variables
        # self.reset() is called by the wrapper/user, so we don't call it here
        # to avoid double-initialization. Variables are set to None.
        
        # Run validation at the end of init
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG from seed
        self.np_random = np.random.default_rng(seed=self.np_random.integers(2**31) if seed is None else seed)

        # Initialize all game state
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.TILE_UNBURNT, dtype=np.int8)
        
        # Place initial 2x2 fire near the center
        start_x = self.np_random.integers(low=3, high=self.GRID_SIZE - 4)
        start_y = self.np_random.integers(low=3, high=self.GRID_SIZE - 4)
        self.grid[start_y:start_y+2, start_x:start_x+2] = self.TILE_FIRE
        
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.turns_remaining = self.MAX_TURNS
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        place_firebreak = action[1] == 1  # Boolean
        # shift_held = action[2] == 1  # Unused per brief

        # --- 1. Player Action Phase ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_SIZE - 1)

        if place_firebreak:
            x, y = self.cursor_pos
            if self.grid[y, x] == self.TILE_UNBURNT:
                self.grid[y, x] = self.TILE_FIREBREAK
                # sfx_place_break()

        # --- 2. Fire Spread Phase ---
        next_grid = self.grid.copy()
        fire_locations = np.argwhere(self.grid == self.TILE_FIRE)
        
        for y, x in fire_locations:
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.GRID_SIZE and 0 <= nx < self.GRID_SIZE and self.grid[ny, nx] == self.TILE_UNBURNT:
                    next_grid[ny, nx] = self.TILE_FIRE
                    # sfx_fire_spread()
        self.grid = next_grid

        # --- 3. Update State & Check Termination ---
        self.steps += 1
        self.turns_remaining -= 1
        
        terminated = False
        reward = 0

        edge_fire = np.any(self.grid[0, :] == self.TILE_FIRE) or \
                    np.any(self.grid[-1, :] == self.TILE_FIRE) or \
                    np.any(self.grid[:, 0] == self.TILE_FIRE) or \
                    np.any(self.grid[:, -1] == self.TILE_FIRE)

        if edge_fire:
            terminated = True
            self.game_over = True
            reward = -50.0 # Terminal loss reward
            # sfx_lose()
        elif self.turns_remaining <= 0:
            terminated = True
            self.game_over = True
            reward = 50.0 # Terminal win reward
            # sfx_win()
        else:
            unburnt_tiles = np.count_nonzero(self.grid == self.TILE_UNBURNT)
            reward = unburnt_tiles / 10.0 # Scaled reward to be in [0, 10]

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turns_remaining": self.turns_remaining,
        }

    def _render_game(self):
        # Draw grid lines for visual structure
        for i in range(self.GRID_SIZE + 1):
            x_pos = self.grid_offset_x + i * self.cell_size
            y_pos = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x_pos, self.grid_offset_y), (x_pos, self.grid_offset_y + self.grid_render_size))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.grid_offset_x, y_pos), (self.grid_offset_x + self.grid_render_size, y_pos))

        # Draw tiles
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                tile_state = self.grid[y, x]
                if tile_state == self.TILE_UNBURNT:
                    pygame.draw.rect(self.screen, self.COLOR_UNBURNT, rect)
                elif tile_state == self.TILE_FIREBREAK:
                    pygame.draw.rect(self.screen, self.COLOR_FIREBREAK, rect)
                    pygame.draw.line(self.screen, (0,0,0,50), rect.topleft, rect.bottomright, 2)
                    pygame.draw.line(self.screen, (0,0,0,50), rect.topright, rect.bottomleft, 2)
                elif tile_state == self.TILE_FIRE:
                    # Flickering fire effect
                    base_color = self.np_random.choice([self.COLOR_FIRE_1, self.COLOR_FIRE_2, self.COLOR_FIRE_3])
                    pygame.draw.rect(self.screen, base_color, rect)
                    inner_rect = rect.inflate(-self.cell_size * 0.5, -self.cell_size * 0.5)
                    inner_color = self.np_random.choice([self.COLOR_FIRE_2, self.COLOR_FIRE_3, (255,255,100)])
                    pygame.draw.rect(self.screen, inner_color, inner_rect)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_x * self.cell_size,
            self.grid_offset_y + cursor_y * self.cell_size,
            self.cell_size, self.cell_size
        )
        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 3)

    def _render_ui(self):
        # Turns remaining
        turns_text = self.font_medium.render(f"Turns Left: {self.turns_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (20, 15))

        # Unburnt tiles
        unburnt_count = np.count_nonzero(self.grid == self.TILE_UNBURNT)
        unburnt_text = self.font_medium.render(f"Unburnt: {unburnt_count}", True, self.COLOR_TEXT)
        unburnt_rect = unburnt_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(unburnt_text, unburnt_rect)
        
        if self.game_over:
            msg_text = "FIRE CONTAINED" if self.turns_remaining <= 0 else "CONTAINMENT FAILED"
            msg_color = self.COLOR_WIN if self.turns_remaining <= 0 else self.COLOR_LOSE
            
            text_surf = self.font_large.render(msg_text, True, msg_color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            shadow_surf = self.font_large.render(msg_text, True, (0,0,0))
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 3, self.SCREEN_HEIGHT / 2 + 3))

            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to initialize state before observation test
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset again
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
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Wildfire Containment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset(seed=42)
    print(f"Initial State: {info}")
    
    terminated = False
    running = True
    
    key_to_action = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4
    }

    while running:
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=random.randint(0, 10000))
                    terminated = False
                    print("--- GAME RESET ---")
                    
                if not terminated:
                    action_taken = False
                    if event.key in key_to_action:
                        action[0] = key_to_action[event.key]
                        action_taken = True
                    if event.key == pygame.K_SPACE:
                        action[1] = 1
                        action_taken = True
                    
                    if action_taken:
                        obs, reward, terminated, truncated, info = env.step(action)
                        turn = GameEnv.MAX_TURNS - info['turns_remaining']
                        print(f"Turn: {turn}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        frame = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()