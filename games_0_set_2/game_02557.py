
# Generated: 2025-08-28T05:17:06.424207
# Source Brief: brief_02557.md
# Brief Index: 2557

        
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
        "Controls: Use arrow keys to move. Push all brown crates onto the green targets before the time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, real-time version of Sokoban. Race against the clock to push all the crates onto their designated targets. Plan your moves carefully but quickly!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 32
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2
        
        # Game constants
        self.INITIAL_TIMER = 60.0  # seconds
        self.FPS = 30
        self.MAX_STEPS = int(self.INITIAL_TIMER * self.FPS)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_CRATE = (160, 110, 70)
        self.COLOR_TARGET = (80, 200, 120)
        self.COLOR_TARGET_FILLED = (120, 255, 160)
        self.COLOR_TEXT = (230, 230, 230)

        # State variables (initialized in reset)
        self.player_pos = None
        self.crate_positions = None
        self.target_positions = None
        self.wall_positions = None
        self.target_positions_set = None
        self.wall_positions_set = None
        self.timer = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.level_layout = [
            "WWWWWWWWWWWWWWWW",
            "W              W",
            "W P C T        W",
            "W   W          W",
            "W C WWW T      W",
            "W   W          W",
            "W C   T        W",
            "W              W",
            "W              W",
            "WWWWWWWWWWWWWWWW",
        ]
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.INITIAL_TIMER
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.wall_positions = []
        self.target_positions = []
        self.crate_positions = []
        for y, row in enumerate(self.level_layout):
            for x, char in enumerate(row):
                pos = [x, y]
                if char == 'W':
                    self.wall_positions.append(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char == 'C':
                    self.crate_positions.append(pos)
                elif char == 'T':
                    self.target_positions.append(pos)
        
        # Use sets for faster lookups
        self.wall_positions_set = {tuple(p) for p in self.wall_positions}
        self.target_positions_set = {tuple(p) for p in self.target_positions}

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Unused
        # shift_held = action[2] == 1  # Unused
        
        # --- Update game logic ---
        reward = -0.01  # Small penalty for each step to encourage speed

        # Store pre-move state for reward calculation
        crates_on_target_before = self._count_crates_on_targets()
        
        # Handle player movement and crate pushing
        self._handle_movement(movement)
        
        # Calculate rewards based on state change
        crates_on_target_after = self._count_crates_on_targets()
        
        placed_count = max(0, crates_on_target_after - crates_on_target_before)
        removed_count = max(0, crates_on_target_before - crates_on_target_after)
        
        reward += placed_count * 1.0   # +1 for placing a crate on a target
        reward += removed_count * -0.5 # -0.5 for moving a crate off a target

        # Adjacency reward
        adj_reward = 0
        for crate_pos in self.crate_positions:
            is_adj = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (crate_pos[0] + dx, crate_pos[1] + dy) in self.target_positions_set:
                    is_adj = True
                    break
            if is_adj:
                adj_reward += 0.1
        reward += adj_reward

        # Update timer and steps
        self.timer -= 1.0 / self.FPS
        self.steps += 1
        
        # Check for termination
        victory = self._count_crates_on_targets() == len(self.target_positions)
        timeout = self.timer <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        terminated = victory or timeout or max_steps_reached
        if terminated:
            self.game_over = True
            if victory:
                reward += 100  # Large reward for winning
            elif timeout:
                reward -= 100  # Large penalty for losing

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx == 0 and dy == 0:
            return

        next_player_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

        # Check for wall collision
        if tuple(next_player_pos) in self.wall_positions_set:
            return # Cannot move into a wall

        # Check for crate collision (and pushing)
        crate_idx = self._get_crate_at(next_player_pos)
        if crate_idx is not None:
            next_crate_pos = [next_player_pos[0] + dx, next_player_pos[1] + dy]
            # Check if crate can be pushed
            if tuple(next_crate_pos) not in self.wall_positions_set and self._get_crate_at(next_crate_pos) is None:
                # Push crate and move player
                self.crate_positions[crate_idx] = next_crate_pos
                self.player_pos = next_player_pos
                # sfx: push_crate.wav
        else:
            # No collision, just move player
            self.player_pos = next_player_pos
            # sfx: step.wav

    def _get_crate_at(self, pos):
        for i, crate_pos in enumerate(self.crate_positions):
            if crate_pos[0] == pos[0] and crate_pos[1] == pos[1]:
                return i
        return None

    def _count_crates_on_targets(self):
        count = 0
        crate_pos_set = {tuple(p) for p in self.crate_positions}
        for target_pos in self.target_positions:
            if tuple(target_pos) in crate_pos_set:
                count += 1
        return count

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(self.X_OFFSET + x * self.TILE_SIZE, self.Y_OFFSET + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect)

        # Draw walls
        for x, y in self.wall_positions:
            rect = pygame.Rect(self.X_OFFSET + x * self.TILE_SIZE, self.Y_OFFSET + y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw targets
        crate_pos_set = {tuple(p) for p in self.crate_positions}
        for x, y in self.target_positions:
            center_x = int(self.X_OFFSET + x * self.TILE_SIZE + self.TILE_SIZE / 2)
            center_y = int(self.Y_OFFSET + y * self.TILE_SIZE + self.TILE_SIZE / 2)
            radius = int(self.TILE_SIZE * 0.4)
            is_filled = (x, y) in crate_pos_set
            color = self.COLOR_TARGET_FILLED if is_filled else self.COLOR_TARGET
            
            # Glowing effect for filled targets
            if is_filled:
                # sfx: target_filled.wav
                glow_radius = int(radius * (1.2 + 0.1 * math.sin(self.steps * 0.2)))
                glow_color = (color[0], color[1], color[2], 64)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, glow_color)
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Draw crates
        for x, y in self.crate_positions:
            rect = pygame.Rect(self.X_OFFSET + x * self.TILE_SIZE + 3, self.Y_OFFSET + y * self.TILE_SIZE + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6)
            pygame.draw.rect(self.screen, self.COLOR_CRATE, rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_CRATE), rect, width=2, border_radius=4)

        # Draw player
        x, y = self.player_pos
        rect = pygame.Rect(self.X_OFFSET + x * self.TILE_SIZE + 4, self.Y_OFFSET + y * self.TILE_SIZE + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=3)

    def _render_ui(self):
        # Draw timer bar
        timer_ratio = max(0, self.timer / self.INITIAL_TIMER)
        bar_width = self.WIDTH * timer_ratio
        
        # Color interpolation from green to red
        r = int(255 * (1 - timer_ratio))
        g = int(200 * timer_ratio)
        timer_color = (r, g, 50)

        pygame.draw.rect(self.screen, timer_color, (0, 0, bar_width, 10))
        pygame.draw.rect(self.screen, (255,255,255, 50), (0, 0, self.WIDTH, 10), 1)

        # Draw text info
        crates_on_target = self._count_crates_on_targets()
        total_crates = len(self.target_positions)
        
        info_text = f"Crates: {crates_on_target}/{total_crates}"
        text_surface = self.font_large.render(info_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (15, 15))

        score_text = f"Score: {self.score:.2f}"
        score_surface = self.font_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.WIDTH - 15, 15))
        self.screen.blit(score_surface, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            victory = crates_on_target == total_crates
            end_text = "LEVEL COMPLETE!" if victory else "TIME UP!"
            end_color = self.COLOR_TARGET_FILLED if victory else self.COLOR_PLAYER
            
            end_surface = pygame.font.SysFont("Arial", 50, bold=True).render(end_text, True, end_color)
            end_rect = end_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surface, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "crates_on_target": self._count_crates_on_targets(),
            "total_crates": len(self.target_positions)
        }

    def close(self):
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sokoban Rush")
    clock = pygame.time.Clock()

    print(env.user_guide)
    
    action = np.array([0, 0, 0]) # No-op, release space, release shift
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Get key presses for manual control
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()