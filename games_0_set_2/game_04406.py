
# Generated: 2025-08-28T02:18:24.186838
# Source Brief: brief_04406.md
# Brief Index: 4406

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fast-paced arcade grid-based coin collection game.
    The player must navigate a grid, collecting all coins within a time limit to
    progress through three increasingly difficult stages.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character one cell at a time. "
        "Space and Shift have no effect."
    )

    # User-facing description of the game
    game_description = (
        "Navigate a grid to collect all the gold coins before the timer runs out. "
        "Complete three stages of increasing difficulty to win. Each move costs time!"
    )

    # The game is turn-based, advancing only on action receipt.
    auto_advance = False

    # --- Game Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 20, 12
    CELL_SIZE = 30
    GRID_PIXEL_W = GRID_W * CELL_SIZE
    GRID_PIXEL_H = GRID_H * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_W - GRID_PIXEL_W) // 2
    GRID_OFFSET_Y = (SCREEN_H - GRID_PIXEL_H) // 2

    NUM_COINS = 25
    STAGE_TIME_LIMIT = 60
    MAX_TOTAL_STEPS = STAGE_TIME_LIMIT * 3

    # --- Colors ---
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_HIGHLIGHT = (255, 255, 200)
    COLOR_COIN = (255, 215, 0)
    COLOR_COIN_OUTLINE = (200, 160, 0)
    COLOR_GRID = (50, 50, 70)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    BG_COLORS = {
        1: (20, 20, 80),  # Dark Blue
        2: (20, 80, 20),  # Dark Green
        3: (80, 20, 80),  # Dark Purple
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = [0, 0]
        self.coin_positions = set()
        self.score = 0
        self.stage = 1
        self.stage_timer = 0
        self.steps = 0
        self.game_over = False
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.stage = 1
        self.game_over = False
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the state for the current stage."""
        self.stage_timer = self.STAGE_TIME_LIMIT
        self.player_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        # Generate coin positions with difficulty scaling
        # Stage 1: radius=0, Stage 2: radius=1, Stage 3: radius=2
        dead_zone_radius = self.stage - 1 
        
        possible_coords = []
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                is_player_start = (x == self.player_pos[0] and y == self.player_pos[1])
                # Manhattan distance from player start
                dist_from_start = abs(x - self.player_pos[0]) + abs(y - self.player_pos[1])
                
                if not is_player_start and dist_from_start > dead_zone_radius:
                    possible_coords.append((x, y))

        # Ensure we don't try to sample more coins than available spots
        num_to_sample = min(self.NUM_COINS, len(possible_coords))
        
        indices = self.np_random.choice(len(possible_coords), num_to_sample, replace=False)
        self.coin_positions = {possible_coords[i] for i in indices}

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        
        movement = action[0]
        # space_held and shift_held are unused per the brief
        
        self.stage_timer -= 1
        self.steps += 1
        
        # --- Handle Movement and Penalties ---
        dx, dy = 0, 0
        is_move_action = True
        if movement == 1: dy = -1   # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1  # Right
        else: # No-op
            is_move_action = False
            reward -= 0.2

        if is_move_action:
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy

            if 0 <= new_x < self.GRID_W and 0 <= new_y < self.GRID_H:
                self.player_pos = [new_x, new_y]
                # sfx_move
            else: # Out-of-bounds move
                reward -= 0.2
                # sfx_bonk

        # --- Check for Coin Collection ---
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.coin_positions:
            self.coin_positions.remove(player_pos_tuple)
            self.score += 1
            reward += 1
            # sfx_coin_collect

        # --- Check for Stage Completion ---
        if not self.coin_positions:
            reward += 5
            if self.stage < 3:
                self.stage += 1
                self._setup_stage()
                # sfx_stage_clear
            else: # Game Won
                self.game_over = True
                reward += 50
                # sfx_game_win
        
        # --- Check for Termination by Time ---
        if self.stage_timer <= 0:
            self.game_over = True
            reward -= 50
            # sfx_game_over

        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.BG_COLORS.get(self.stage, (0,0,0)))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Grid
        for i in range(self.GRID_W + 1):
            start = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_PIXEL_H)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_H + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_PIXEL_W, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw Coins
        coin_radius = int(self.CELL_SIZE * 0.3)
        pulse = math.sin(self.steps * 0.4) * 2
        for x, y in self.coin_positions:
            cx = self.GRID_OFFSET_X + int((x + 0.5) * self.CELL_SIZE)
            cy = self.GRID_OFFSET_Y + int((y + 0.5) * self.CELL_SIZE)
            current_radius = max(1, int(coin_radius + pulse))
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, current_radius, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, current_radius, self.COLOR_COIN_OUTLINE)

        # Draw Player
        px, py = self.player_pos
        bob = math.sin(self.steps * 0.5) * 3
        player_rect = pygame.Rect(
            self.GRID_OFFSET_X + px * self.CELL_SIZE + 4,
            self.GRID_OFFSET_Y + py * self.CELL_SIZE + 4 - int(bob),
            self.CELL_SIZE - 8,
            self.CELL_SIZE - 8
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HIGHLIGHT, player_rect, width=2, border_radius=4)

    def _render_text(self, text, font, x, y, color, shadow_color, align="topleft"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        shadow_rect = shadow_surf.get_rect()

        if align == "topleft":
            text_rect.topleft = (x, y)
            shadow_rect.topleft = (x + 2, y + 2)
        elif align == "topright":
            text_rect.topright = (x, y)
            shadow_rect.topright = (x + 2, y + 2)
        elif align == "center":
            text_rect.center = (x, y)
            shadow_rect.center = (x + 2, y + 2)
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", self.font_ui, 10, 10, 
                          self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="topleft")
        
        # Timer
        self._render_text(f"TIME: {self.stage_timer}", self.font_ui, self.SCREEN_W - 10, 10, 
                          self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="topright")
        
        # Stage
        self._render_text(f"STAGE {self.stage}", self.font_ui, self.SCREEN_W // 2, self.SCREEN_H - 20, 
                          self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="center")

        # Game Over / Win Message
        if self.game_over:
            is_win = not self.coin_positions and self.stage == 3
            message = "YOU WIN!" if is_win else "GAME OVER"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            self._render_text(message, self.font_msg, self.SCREEN_W // 2, self.SCREEN_H // 2, 
                              color, self.COLOR_TEXT_SHADOW, align="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "stage_timer": self.stage_timer,
            "coins_left": len(self.coin_positions),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

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
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11" or "windows" or "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Coin Collector")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        movement = 0 # No-op default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit
                    terminated = True
        
        # Action is taken on key press, not key hold for this turn-based game
        if movement != 0 or terminated:
            action = [movement, 0, 0] # Space/Shift are not used
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    env.close()