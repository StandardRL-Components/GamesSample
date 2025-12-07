
# Generated: 2025-08-28T03:15:23.654713
# Source Brief: brief_04868.md
# Brief Index: 4868

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move your character and push the crates."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced puzzle game where you must push all crates onto their targets before time runs out."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    GRID_SIZE = 40
    
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    
    # Colors (Bright and contrasting)
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (35, 35, 50)
    COLOR_WALL = (60, 60, 80)
    COLOR_WALL_BORDER = (80, 80, 100)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_BORDER = (200, 150, 0)
    COLOR_CRATE = (160, 82, 45)
    COLOR_CRATE_BORDER = (120, 60, 30)
    COLOR_TARGET = (50, 150, 50)
    COLOR_TARGET_INNER = (80, 200, 80)
    COLOR_CRATE_ON_TARGET = (150, 180, 80)
    
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_TIME_WARN = (255, 50, 50)
    COLOR_UI_WIN = (100, 255, 100)
    COLOR_UI_LOSE = (255, 100, 100)

    # Level Layout (W=Wall, P=Player, C=Crate, T=Target, .=Empty)
    LEVEL_MAP = [
        "WWWWWWWWWWWWWWWW",
        "W.T............W",
        "W...W......W.C.W",
        "W.C.W..P...W...W",
        "W..............W",
        "W...WWWWWWWW.T.W",
        "W.T..........C.W",
        "W...W......W...W",
        "W.C.W..T...W.C.W",
        "WWWWWWWWWWWWWWWW",
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = 0.0
        self.player_pos = None
        self.crate_pos = None
        self.target_pos = None
        self.wall_pos = None
        self.num_crates = 0
        self.max_steps = self.FPS * self.TIME_LIMIT_SECONDS

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS
        
        # Parse level map
        self.wall_pos = set()
        initial_crate_pos = []
        self.target_pos = []
        for r, row_str in enumerate(self.LEVEL_MAP):
            for c, char in enumerate(row_str):
                pos = (c, r)
                if char == 'W':
                    self.wall_pos.add(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char == 'C':
                    initial_crate_pos.append(pos)
                elif char == 'T':
                    self.target_pos.append(pos)
        
        # Ensure consistent pairing of crates and targets
        self.crate_pos = sorted(initial_crate_pos)
        self.target_pos.sort()
        self.num_crates = len(self.crate_pos)

        # Pre-calculate initial state for reward
        self.last_total_dist = self._calculate_total_dist()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_left -= 1 / self.FPS
        
        # Store pre-move state for reward calculation
        old_crate_pos = list(self.crate_pos)
        old_on_target_indices = self._get_on_target_indices()

        # Handle movement and collisions
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            # Check for wall collision
            if next_player_pos in self.wall_pos:
                pass # Player is blocked by wall
            
            # Check for crate collision (and push)
            elif next_player_pos in self.crate_pos:
                crate_index = self.crate_pos.index(next_player_pos)
                next_crate_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                
                # Check if crate push is blocked by a wall or another crate
                if next_crate_pos not in self.wall_pos and next_crate_pos not in self.crate_pos:
                    # Move player and crate
                    self.player_pos = next_player_pos
                    self.crate_pos[crate_index] = next_crate_pos
                    # sfx: Crate push sound
            
            # Free movement
            else:
                self.player_pos = next_player_pos
                # sfx: Player step sound

        # --- Calculate Reward ---
        reward = self._calculate_reward(old_crate_pos, old_on_target_indices)
        self.score += reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            # Add terminal reward/penalty
            if self._all_crates_on_target():
                self.score += 100
                reward += 100
            else:
                self.score -= 100
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_reward(self, old_crate_pos, old_on_target_indices):
        reward = -0.01  # Small penalty for each step

        # Reward for moving crates closer to their targets
        new_total_dist = self._calculate_total_dist()
        dist_delta = self.last_total_dist - new_total_dist
        reward += dist_delta * 0.1
        self.last_total_dist = new_total_dist

        # Reward for placing a crate on a target
        new_on_target_indices = self._get_on_target_indices()
        newly_placed = len(new_on_target_indices - old_on_target_indices)
        if newly_placed > 0:
            reward += newly_placed * 10
            # sfx: Success chime
        
        return reward

    def _check_termination(self):
        if self._all_crates_on_target():
            return True
        if self.time_left <= 0:
            return True
        if self.steps >= self.max_steps:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines for background texture
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        for pos in self.target_pos:
            rect = pygame.Rect(pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TARGET, rect)
            pygame.draw.rect(self.screen, self.COLOR_TARGET_INNER, rect.inflate(-8, -8))

        # Draw walls
        for pos in self.wall_pos:
            rect = pygame.Rect(pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_WALL_BORDER, rect, 2)

        # Draw crates
        on_target_indices = self._get_on_target_indices()
        for i, pos in enumerate(self.crate_pos):
            rect = pygame.Rect(pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            color = self.COLOR_CRATE_ON_TARGET if i in on_target_indices else self.COLOR_CRATE
            border_color = self.COLOR_CRATE_BORDER
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
            pygame.draw.rect(self.screen, border_color, rect.inflate(-4, -4), 3)

        # Draw player
        player_center = (
            int(self.player_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2),
            int(self.player_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
        )
        radius = int(self.GRID_SIZE / 2 - 4)
        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center[0], player_center[1], radius, self.COLOR_PLAYER_BORDER)

    def _render_ui(self):
        # Render crates on target count
        num_on_target = len(self._get_on_target_indices())
        crates_text = f"CRATES: {num_on_target} / {self.num_crates}"
        text_surf = self.font_ui.render(crates_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, self.SCREEN_HEIGHT - 30))

        # Render timer
        time_color = self.COLOR_UI_TIME_WARN if self.time_left < 10 and self.steps % self.FPS > self.FPS / 2 else self.COLOR_UI_TEXT
        minutes = int(self.time_left) // 60
        seconds = int(self.time_left) % 60
        time_text = f"TIME: {minutes:02}:{seconds:02}"
        text_surf = self.font_ui.render(time_text, True, time_color)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(text_surf, text_rect)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            if self._all_crates_on_target():
                msg = "LEVEL CLEAR!"
                color = self.COLOR_UI_WIN
            else:
                msg = "TIME UP!"
                color = self.COLOR_UI_LOSE
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "crates_on_target": len(self._get_on_target_indices()),
        }
        
    def _calculate_total_dist(self):
        # Manhattan distance between each crate and its corresponding target
        return sum(
            abs(c[0] - t[0]) + abs(c[1] - t[1])
            for c, t in zip(self.crate_pos, self.target_pos)
        )

    def _get_on_target_indices(self):
        return {i for i, (c, t) in enumerate(zip(self.crate_pos, self.target_pos)) if c == t}

    def _all_crates_on_target(self):
        return len(self._get_on_target_indices()) == self.num_crates

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Arcade")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0

    print("\n" + "="*30)
    print(f"GAME: Sokoban Arcade")
    print(f"INFO: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while running:
        movement_action = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        # Construct the MultiDiscrete action
        action = [movement_action, 0, 0] # space and shift are not used

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()