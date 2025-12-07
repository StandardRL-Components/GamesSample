
# Generated: 2025-08-27T13:01:39.399297
# Source Brief: brief_00234.md
# Brief Index: 234

        
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
        "Controls: Use arrow keys (↑↓←→) to move your player (red circle). "
        "Push crates (brown squares) onto the green targets."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A fast-paced, time-based puzzle game. Race against the clock to push all crates onto their targets."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TILE_SIZE = 32
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        
        # Game constants
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.FPS * self.GAME_DURATION_SECONDS + 10 # Safety buffer
        self.INTERP_SPEED = 0.3  # Speed for smooth visual movement
        self.PUSH_FLASH_DURATION = 5 # frames

        # Colors
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_WALL = (127, 140, 141)
        self.COLOR_PLAYER = (231, 76, 60)
        self.COLOR_CRATE = (189, 126, 47)
        self.COLOR_TARGET = (46, 204, 113)
        self.COLOR_CRATE_ON_TARGET = (26, 188, 156)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_FLASH = (255, 255, 255)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_timer = pygame.font.Font(None, 48)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game State Initialization ---
        self.level_map_str = [
            "WWWWWWWWWWWWWWWWWWWW",
            "W        W         W",
            "W P C    W   T     W",
            "W W W    W         W",
            "W   C  WWWWWWWW    W",
            "W        W    C T  W",
            "W        W         W",
            "W   T              W",
            "W                  W",
            "WWWWWWWWWWWWWWWWWWWW",
            "....................",
            "....................",
        ]

        self.player_log_pos = None
        self.player_vis_pos = None
        self.crates_log_pos = None
        self.crates_vis_pos = None
        self.targets_pos = None
        self.walls = None
        self.crate_flash_timers = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.last_crates_on_target = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.FPS * self.GAME_DURATION_SECONDS
        
        self._parse_level()

        self.player_vis_pos = [self.player_log_pos[0] * self.TILE_SIZE, self.player_log_pos[1] * self.TILE_SIZE]
        self.crates_vis_pos = [[x * self.TILE_SIZE, y * self.TILE_SIZE] for x, y in self.crates_log_pos]
        self.crate_flash_timers = [0] * len(self.crates_log_pos)
        self.last_crates_on_target = self._count_crates_on_target()
        
        return self._get_observation(), self._get_info()

    def _parse_level(self):
        self.player_log_pos = (0, 0)
        self.crates_log_pos = []
        self.targets_pos = []
        self.walls = []
        for y, row in enumerate(self.level_map_str):
            for x, char in enumerate(row):
                if char == 'W':
                    self.walls.append((x, y))
                elif char == 'P':
                    self.player_log_pos = (x, y)
                elif char == 'C':
                    self.crates_log_pos.append((x, y))
                elif char == 'T':
                    self.targets_pos.append((x, y))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Time penalty

        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1)

        # Update flash timers
        for i in range(len(self.crate_flash_timers)):
            self.crate_flash_timers[i] = max(0, self.crate_flash_timers[i] - 1)

        # Handle player movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            next_player_pos = (self.player_log_pos[0] + dx, self.player_log_pos[1] + dy)

            # Check for wall collision
            if next_player_pos in self.walls:
                # Placeholder for bump sound/effect
                # sfx: player_bump_wall.wav
                pass
            
            # Check for crate collision
            elif next_player_pos in self.crates_log_pos:
                crate_index = self.crates_log_pos.index(next_player_pos)
                next_crate_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)

                # Check if space behind crate is clear
                if next_crate_pos not in self.walls and next_crate_pos not in self.crates_log_pos:
                    # Push crate
                    self.crates_log_pos[crate_index] = next_crate_pos
                    self.player_log_pos = next_player_pos
                    self.crate_flash_timers[crate_index] = self.PUSH_FLASH_DURATION
                    # sfx: crate_push.wav
            
            # No collision, move player
            else:
                self.player_log_pos = next_player_pos
                # sfx: player_step.wav

        # --- Calculate Rewards ---
        current_crates_on_target = self._count_crates_on_target()
        if current_crates_on_target > self.last_crates_on_target:
            reward += 1.0 * (current_crates_on_target - self.last_crates_on_target)
            # sfx: crate_on_target.wav
        self.last_crates_on_target = current_crates_on_target
        self.score += reward

        # --- Check Termination ---
        terminated = False
        all_crates_on_targets = self._count_crates_on_target() == len(self.targets_pos)

        if all_crates_on_targets:
            reward += 100.0
            self.score += 100.0
            terminated = True
            # sfx: level_complete.wav
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100.0
            self.score -= 100.0
            terminated = True
            # sfx: time_up.wav

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _count_crates_on_target(self):
        return sum(1 for crate_pos in self.crates_log_pos if crate_pos in self.targets_pos)

    def _get_observation(self):
        # --- Update Visuals (Interpolation) ---
        # Player
        target_vis_x = self.player_log_pos[0] * self.TILE_SIZE
        target_vis_y = self.player_log_pos[1] * self.TILE_SIZE
        self.player_vis_pos[0] += (target_vis_x - self.player_vis_pos[0]) * self.INTERP_SPEED
        self.player_vis_pos[1] += (target_vis_y - self.player_vis_pos[1]) * self.INTERP_SPEED

        # Crates
        for i in range(len(self.crates_log_pos)):
            target_vis_x = self.crates_log_pos[i][0] * self.TILE_SIZE
            target_vis_y = self.crates_log_pos[i][1] * self.TILE_SIZE
            self.crates_vis_pos[i][0] += (target_vis_x - self.crates_vis_pos[i][0]) * self.INTERP_SPEED
            self.crates_vis_pos[i][1] += (target_vis_y - self.crates_vis_pos[i][1]) * self.INTERP_SPEED

        # --- Render Game ---
        self.screen.fill(self.COLOR_BG)
        
        # Draw Targets
        for x, y in self.targets_pos:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TARGET, rect)

        # Draw Walls
        for x, y in self.walls:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw Crates
        for i, (log_pos, vis_pos) in enumerate(zip(self.crates_log_pos, self.crates_vis_pos)):
            rect = pygame.Rect(vis_pos[0], vis_pos[1], self.TILE_SIZE, self.TILE_SIZE)
            color = self.COLOR_CRATE
            if log_pos in self.targets_pos:
                color = self.COLOR_CRATE_ON_TARGET
            if self.crate_flash_timers[i] > 0:
                color = self.COLOR_FLASH
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2) # Border

        # Draw Player
        px, py = self.player_vis_pos
        center_x = int(px + self.TILE_SIZE / 2)
        center_y = int(py + self.TILE_SIZE / 2)
        radius = int(self.TILE_SIZE / 2 * 0.8)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Crates on target count
        crates_placed_text = f"Placed: {self._count_crates_on_target()}/{len(self.targets_pos)}"
        text_surf = self.font_ui.render(crates_placed_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, self.HEIGHT - 30))

        # Timer
        time_seconds = self.time_remaining // self.FPS
        time_str = f"{time_seconds:02d}"
        timer_surf = self.font_timer.render(time_str, True, self.COLOR_TEXT)
        timer_rect = timer_surf.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(timer_surf, timer_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = self._count_crates_on_target() == len(self.targets_pos)
            msg = "LEVEL CLEAR" if win_condition else "TIME UP"
            color = self.COLOR_TARGET if win_condition else self.COLOR_PLAYER
            
            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining // self.FPS,
            "crates_on_target": self._count_crates_on_target()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to run in a window
    import os
    os.environ.pop('SDL_VIDEODRIVER', None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Re-initialize Pygame for display
    pygame.display.set_caption("Sokoban Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op
    action[1] = 0
    action[2] = 0

    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Event Handling (get keys for manual play) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # No movement
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # Convert the observation back to a Pygame Surface
        # The observation is (H, W, C), but pygame wants (W, H, C)
        # We need to transpose it back from (H, W, C) to (W, H, C) for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()