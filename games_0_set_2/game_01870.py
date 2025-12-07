
# Generated: 2025-08-27T18:33:52.688612
# Source Brief: brief_01870.md
# Brief Index: 1870

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block, ↓ to speed up its descent."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Align 3 or more blocks of the same color to clear them. Clear 15 blocks to win before the grid fills up!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
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
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 15
        self.BLOCK_SIZE = 24
        self.GRID_X_OFFSET = (self.screen_width - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_Y_OFFSET = (self.screen_height - self.GRID_HEIGHT * self.BLOCK_SIZE) + 20
        self.MAX_STEPS = 2000
        self.WIN_CONDITION_CLEARED = 15
        self.ANIMATION_DURATION = 6 # frames (1/5th of a second at 30fps)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 230)
        
        self.BLOCK_COLORS = {
            1: (255, 80, 80),   # Red
            2: (80, 255, 80),   # Green
            3: (80, 120, 255),  # Blue
            4: (128, 128, 128)  # Grey (obstacle)
        }
        self.BLOCK_BORDERS = {k: tuple(max(0, c-40) for c in v) for k, v in self.BLOCK_COLORS.items()}

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.np_random = None
        self.grid = None
        self.falling_block = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.cleared_blocks_count = None
        self.fall_speed = None
        self.animation_state = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cleared_blocks_count = 0
        self.fall_speed = 0.5
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.animation_state = {'clearing': set(), 'timer': 0}

        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(30)
        self.steps += 1
        reward = 0

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Game Logic ---
        
        # Phase 1: Animation is active
        if self.animation_state['timer'] > 0:
            self.animation_state['timer'] -= 1
            if self.animation_state['timer'] == 0:
                reward += self._process_cleared_blocks()
        
        # Phase 2: Normal gameplay (no animation)
        else:
            if self.falling_block:
                # Handle input
                self._handle_input(movement)
                
                # Update block falling
                fall_boost = 2.0 if movement == 2 else 1.0 # Down arrow speeds up fall
                self.falling_block['y'] += self.fall_speed * fall_boost / 30.0 # Scale speed by FPS

                # Check for landing
                if self._check_collision(self.falling_block['x'], int(self.falling_block['y'] + 1)):
                    self._place_block_on_grid()
                    match_reward, matched_blocks = self._find_and_process_matches()
                    reward += match_reward
                    
                    if not matched_blocks:
                        reward -= 0.01 # Penalty for non-matching move
                        self._spawn_block()
                    else:
                        # Start animation
                        self.animation_state['clearing'] = matched_blocks
                        self.animation_state['timer'] = self.ANIMATION_DURATION

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.fall_speed = min(2.0, self.fall_speed + 0.05)
            
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.cleared_blocks_count >= self.WIN_CONDITION_CLEARED:
                reward += 100  # Win
            else:
                reward += -100 # Loss (grid full or max steps)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if not self.falling_block: return
        
        dx = 0
        if movement == 3: dx = -1 # Left
        if movement == 4: dx = 1  # Right
        
        if dx != 0:
            new_x = self.falling_block['x'] + dx
            if not self._check_collision(new_x, int(self.falling_block['y'])):
                self.falling_block['x'] = new_x

    def _process_cleared_blocks(self):
        """Actually removes blocks, applies gravity, and checks for chains."""
        reward = 0
        # 1. Remove blocks
        for r, c in self.animation_state['clearing']:
            self.grid[r, c] = 0
        self.animation_state['clearing'] = set()
        
        # 2. Apply gravity
        self._apply_gravity()
        
        # 3. Check for chain reactions
        chain_reward, matched_blocks = self._find_and_process_matches()
        reward += chain_reward
        if matched_blocks:
            # Start a new animation for the chain
            self.animation_state['clearing'] = matched_blocks
            self.animation_state['timer'] = self.ANIMATION_DURATION
        else:
            # No chain, spawn the next block
            self._spawn_block()
        
        return reward

    def _find_and_process_matches(self):
        matched_blocks = self._find_all_matches()
        if not matched_blocks:
            return 0, set()

        reward = 0
        num_cleared = len(matched_blocks)
        self.cleared_blocks_count += num_cleared
        
        # Event-based rewards
        if num_cleared == 3: reward += 1
        elif num_cleared == 4: reward += 2
        elif num_cleared >= 5: reward += 3
        
        # Continuous feedback reward
        reward += num_cleared * 0.1
        
        self.score += reward
        return reward, matched_blocks

    def _find_all_matches(self):
        to_clear = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = self.grid[r, c]
                if color == 0 or color == 4: continue # Skip empty or grey blocks

                # Horizontal check
                if c + 2 < self.GRID_WIDTH and self.grid[r, c+1] == color and self.grid[r, c+2] == color:
                    h_run = [(r, c + i) for i in range(self.GRID_WIDTH - c) if self.grid[r, c+i] == color]
                    if len(h_run) >= 3: to_clear.update(h_run)

                # Vertical check
                if r + 2 < self.GRID_HEIGHT and self.grid[r+1, c] == color and self.grid[r+2, c] == color:
                    v_run = [(r + i, c) for i in range(self.GRID_HEIGHT - r) if self.grid[r+i, c] == color]
                    if len(v_run) >= 3: to_clear.update(v_run)
        return to_clear
        
    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            col = self.grid[:, c]
            non_zeros = col[col != 0]
            new_col = np.zeros(self.GRID_HEIGHT, dtype=int)
            new_col[self.GRID_HEIGHT - len(non_zeros):] = non_zeros
            self.grid[:, c] = new_col

    def _check_collision(self, x, y):
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return True # Out of bounds
        if self.grid[y, x] != 0:
            return True # Collides with another block
        return False

    def _place_block_on_grid(self):
        if not self.falling_block: return
        r, c = int(self.falling_block['y']), self.falling_block['x']
        if 0 <= r < self.GRID_HEIGHT and 0 <= c < self.GRID_WIDTH:
            self.grid[r, c] = self.falling_block['color']
        self.falling_block = None

    def _spawn_block(self):
        # 10% chance of a grey obstacle block
        color_id = 4 if self.np_random.random() < 0.1 else self.np_random.integers(1, 4)
        
        spawn_x = self.np_random.integers(0, self.GRID_WIDTH)
        
        if self.grid[0, spawn_x] != 0:
            self.game_over = True
            self.falling_block = None
        else:
            self.falling_block = {
                'color': color_id,
                'x': spawn_x,
                'y': 0.0
            }

    def _check_termination(self):
        win = self.cleared_blocks_count >= self.WIN_CONDITION_CLEARED
        loss = self.game_over # Set by spawn failure
        timeout = self.steps >= self.MAX_STEPS
        return win or loss or timeout

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cleared_blocks": self.cleared_blocks_count
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, 
                                self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw landed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_id = self.grid[r, c]
                if color_id != 0:
                    is_clearing = (r, c) in self.animation_state['clearing']
                    self._draw_block(c, r, color_id, is_clearing)
        
        # Draw falling block
        if self.falling_block:
            self._draw_block(self.falling_block['x'], self.falling_block['y'], self.falling_block['color'])
    
    def _draw_block(self, c, r, color_id, is_clearing=False):
        x_pos = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
        y_pos = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
        
        rect = pygame.Rect(x_pos, int(y_pos), self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        color = self.BLOCK_COLORS[color_id]
        border_color = self.BLOCK_BORDERS[color_id]
        
        if is_clearing and self.animation_state['timer'] % 2 == 0:
            # Flash white for animation
            color = (255, 255, 255)
            border_color = (200, 200, 200)

        pygame.draw.rect(self.screen, border_color, rect)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, color, inner_rect)
        
    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Cleared blocks display
        cleared_text = self.font_large.render(f"CLEARED: {self.cleared_blocks_count} / {self.WIN_CONDITION_CLEARED}", True, self.COLOR_UI_TEXT)
        cleared_rect = cleared_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(cleared_text, cleared_rect)

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Note: Gymnasium's 'human' render mode is not used here; we manually handle rendering.
    import os
    # Set a dummy video driver to run pygame headlessly if not playing interactively
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    running = True
    total_reward = 0
    
    # Action state
    action = env.action_space.sample()
    action.fill(0)

    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get key presses for action
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}, Cleared: {info['cleared_blocks']}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the screen
        # Need to transpose the observation back to pygame's (width, height, channels) format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise Exception
            display_surf.blit(surf, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode((env.screen_width, env.screen_height))
            display_surf.blit(surf, (0, 0))
            
        pygame.display.flip()

    pygame.quit()