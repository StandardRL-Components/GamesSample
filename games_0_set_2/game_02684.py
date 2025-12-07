
# Generated: 2025-08-27T21:07:49.737306
# Source Brief: brief_02684.md
# Brief Index: 2684

        
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
        "Controls: Use arrow keys (↑↓←→) to move the red player. Push the blue blocks onto the green target squares."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced block-pushing puzzle. Race against the clock to move all the blocks to their designated targets."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE
        
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
        self.FONT_UI = pygame.font.SysFont("monospace", 22, bold=True)
        self.FONT_TIMER = pygame.font.SysFont("monospace", 30, bold=True)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (255, 70, 70)
        self.COLOR_PLAYER_BORDER = (255, 150, 150)
        self.COLOR_BLOCK = (70, 120, 255)
        self.COLOR_BLOCK_BORDER = (150, 180, 255)
        self.COLOR_TARGET = (70, 180, 70)
        self.COLOR_BLOCK_ON_TARGET = (120, 200, 255) # Brighter blue
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER_WARN = (255, 200, 0)
        self.COLOR_TIMER_CRIT = (255, 0, 0)

        # Game constants
        self.INITIAL_TIME = 60.0
        self.FPS = 30 # For smooth visuals and consistent timing
        self.MAX_STEPS = int(self.INITIAL_TIME * self.FPS)
        self.MOVE_COOLDOWN_FRAMES = 5 # Prevents hyper-speed movement
        
        # Initialize state variables
        self.player_pos = None
        self.block_positions = None
        self.target_positions = None
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.move_cooldown = 0
        self.blocks_on_target_last_step = 0
        self.np_random = None

        # This will call reset() and initialize the state
        self.reset()

        # Final check
        self.validate_implementation()
    
    def _generate_level(self):
        """Generates a new random level layout."""
        num_blocks = self.np_random.integers(3, 5)
        
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_cells)
        
        # Ensure player is not on an edge for easier start
        player_candidates = [(x, y) for x, y in all_cells if 1 < x < self.GRID_WIDTH - 2 and 1 < y < self.GRID_HEIGHT - 2]
        if not player_candidates: # Fallback if grid is too small
             player_candidates = all_cells
        
        self.player_pos = player_candidates[0]
        
        # Remove used cell
        all_cells.remove(self.player_pos)
        
        # Place targets and blocks
        self.target_positions = [all_cells.pop(0) for _ in range(num_blocks)]
        self.block_positions = [all_cells.pop(0) for _ in range(num_blocks)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.INITIAL_TIME
        self.move_cooldown = 0
        self.blocks_on_target_last_step = self._count_blocks_on_target()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Time and Step Progression ---
        self.time_remaining -= 1.0 / self.FPS
        self.steps += 1
        
        # --- Base Reward ---
        reward = -0.01  # Small penalty for each step to encourage speed

        # --- Action Processing ---
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Cooldown for player movement to make it human-playable
        self.move_cooldown = max(0, self.move_cooldown - 1)

        if movement != 0 and self.move_cooldown == 0:
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            
            player_next_x, player_next_y = self.player_pos[0] + dx, self.player_pos[1] + dy

            # Check for block collision
            if (player_next_x, player_next_y) in self.block_positions:
                block_idx = self.block_positions.index((player_next_x, player_next_y))
                block_next_x, block_next_y = player_next_x + dx, player_next_y + dy
                
                # Check if block push is valid (within bounds and not hitting another block)
                is_in_bounds = (0 <= block_next_x < self.GRID_WIDTH and 0 <= block_next_y < self.GRID_HEIGHT)
                is_clear = (block_next_x, block_next_y) not in self.block_positions
                
                if is_in_bounds and is_clear:
                    # Push block
                    self.block_positions[block_idx] = (block_next_x, block_next_y)
                    # Move player
                    self.player_pos = (player_next_x, player_next_y)
                    # Sound placeholder: # sfx_push_block()
            
            # Check for wall collision
            elif (0 <= player_next_x < self.GRID_WIDTH and 0 <= player_next_y < self.GRID_HEIGHT):
                # Move player into empty space
                self.player_pos = (player_next_x, player_next_y)
                # Sound placeholder: # sfx_player_move()

        # --- Reward for Block Placement ---
        num_on_target_now = self._count_blocks_on_target()
        reward_change = num_on_target_now - self.blocks_on_target_last_step
        if reward_change > 0:
            reward += reward_change * 1.0  # +1 for each new block on target
            # Sound placeholder: # sfx_target_achieved()
        self.blocks_on_target_last_step = num_on_target_now

        # --- Termination Check ---
        terminated = False
        win = num_on_target_now == len(self.target_positions)
        timeout = self.time_remaining <= 0 or self.steps >= self.MAX_STEPS

        if win:
            reward += 50.0
            terminated = True
            self.game_over = True
            # Sound placeholder: # sfx_win_game()
        elif timeout:
            reward -= 50.0
            terminated = True
            self.game_over = True
            # Sound placeholder: # sfx_lose_game()
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _count_blocks_on_target(self):
        """Counts how many blocks are currently on a target."""
        return len(set(self.block_positions) & set(self.target_positions))
    
    def _draw_pixel_art_square(self, pos, color, border_color):
        """Draws a square with a border for a pixelated look."""
        x, y = pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE
        # Draw border
        pygame.draw.rect(self.screen, border_color, (x, y, self.CELL_SIZE, self.CELL_SIZE))
        # Draw inner fill
        pygame.draw.rect(self.screen, color, (x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw targets
        for pos in self.target_positions:
            x, y = pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE
            pygame.gfxdraw.box(self.screen, (x, y, self.CELL_SIZE, self.CELL_SIZE), self.COLOR_TARGET)
        
        # Draw blocks
        for i, pos in enumerate(self.block_positions):
            if pos in self.target_positions:
                self._draw_pixel_art_square(pos, self.COLOR_BLOCK_ON_TARGET, self.COLOR_BLOCK_BORDER)
            else:
                self._draw_pixel_art_square(pos, self.COLOR_BLOCK, self.COLOR_BLOCK_BORDER)

        # Draw player
        self._draw_pixel_art_square(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_BORDER)

    def _render_ui(self):
        # --- Render Block Count ---
        block_count_text = f"Targets: {self.blocks_on_target_last_step}/{len(self.target_positions)}"
        text_surf = self.FONT_UI.render(block_count_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # --- Render Timer ---
        time_str = f"{max(0, self.time_remaining):.1f}"
        timer_color = self.COLOR_TEXT
        if self.time_remaining < 10:
            timer_color = self.COLOR_TIMER_CRIT
        elif self.time_remaining < 30:
            timer_color = self.COLOR_TIMER_WARN
        
        timer_surf = self.FONT_TIMER.render(time_str, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 7))
        self.screen.blit(timer_surf, timer_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "blocks_on_target": self.blocks_on_target_last_step,
            "total_blocks": len(self.target_positions),
        }

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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a separate display for human play
    pygame.display.set_caption("Block Pusher")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    
    while not terminated:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
        keys = pygame.key.get_pressed()
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

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            print(f"Reason: {'Victory!' if info['blocks_on_target'] == info['total_blocks'] else 'Timeout'}")

    env.close()