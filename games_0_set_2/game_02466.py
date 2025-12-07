
# Generated: 2025-08-27T20:27:36.806223
# Source Brief: brief_02466.md
# Brief Index: 2466

        
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
    """
    A Gymnasium environment for a fast-paced arcade puzzle game.
    The player must navigate a grid and "eat" numbers in ascending order (1 to 20)
    against a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your blue square across the grid. "
        "Your goal is to touch the numbers in ascending order (1, 2, 3...) as fast as possible."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced arcade puzzle game. Navigate a grid to 'eat' numbers in ascending order "
        "from 1 to 20. Beat the 10-second mark for a huge score bonus, but watch out for the 20-second time limit!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 16
        self.GRID_ROWS = 10
        self.CELL_SIZE = self.SCREEN_WIDTH // self.GRID_COLS
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 20
        self.WIN_TIME_SECONDS = 10
        self.NUM_TARGETS = 20
        self.MOVE_COOLDOWN_FRAMES = 3  # Player can move once every 3 frames

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_OUTLINE = (180, 220, 255)
        self.COLOR_TEXT_UI = (220, 220, 240)
        self.COLOR_NUMBER_BASE = (150, 150, 170)
        self.COLOR_NUMBER_MAX = (255, 255, 255)
        self.COLOR_FLASH = (255, 255, 120)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Fonts ---
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_number = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
            self.font_number = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # --- State Variables ---
        self.rng = None
        self.player_pos = None
        self.numbers = []
        self.flash_effects = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = 0
        self.current_target_number = 1
        self.numbers_eaten = 0
        self.move_cooldown = 0
        self.last_dist_to_target = 0

        # Initialize state and validate
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.numbers_eaten = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.current_target_number = 1
        self.move_cooldown = 0
        self.flash_effects = []

        # Generate board layout
        all_cells = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.rng.shuffle(all_cells)

        self.player_pos = all_cells.pop(0)

        self.numbers = []
        for i in range(self.NUM_TARGETS):
            self.numbers.append({'pos': all_cells.pop(0), 'val': i + 1})

        self.last_dist_to_target = self._get_dist_to_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            movement = action[0]
            self._update_timers()
            self._handle_movement(movement)
            reward += self._check_consumption()
            reward += self._calculate_distance_reward()
            terminated, terminal_reward = self._check_termination()
            reward += terminal_reward

        self.steps += 1
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_timers(self):
        self.time_remaining = max(0, self.time_remaining - 1)
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        # Update flash effects
        for flash in self.flash_effects:
            flash['timer'] -= 1
        self.flash_effects = [f for f in self.flash_effects if f['timer'] > 0]

    def _handle_movement(self, movement):
        if self.move_cooldown <= 0 and movement != 0:
            px, py = self.player_pos
            if movement == 1: py -= 1  # Up
            elif movement == 2: py += 1  # Down
            elif movement == 3: px -= 1  # Left
            elif movement == 4: px += 1  # Right

            # Clamp to grid boundaries
            px = max(0, min(self.GRID_COLS - 1, px))
            py = max(0, min(self.GRID_ROWS - 1, py))

            if (px, py) != self.player_pos:
                self.player_pos = (px, py)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
                # sfx: player_move

    def _check_consumption(self):
        for i, num in enumerate(self.numbers):
            if num['pos'] == self.player_pos and num['val'] == self.current_target_number:
                self.numbers_eaten += 1
                self.current_target_number += 1
                self.flash_effects.append({'pos': num['pos'], 'timer': 10}) # 1/3 second flash
                self.numbers.pop(i)
                self.last_dist_to_target = self._get_dist_to_target()
                # sfx: correct_number
                return 1.0  # Event-based reward for correct consumption
        return 0.0

    def _calculate_distance_reward(self):
        reward = 0.0
        current_dist = self._get_dist_to_target()
        if self.current_target_number <= self.NUM_TARGETS:
            if current_dist < self.last_dist_to_target:
                reward += 0.1
            elif current_dist > self.last_dist_to_target:
                reward -= 0.1
        self.last_dist_to_target = current_dist
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0
        if self.numbers_eaten == self.NUM_TARGETS:
            self.game_over = True
            self.win = True
            terminated = True
            time_taken_seconds = (self.TIME_LIMIT_SECONDS * self.FPS - self.time_remaining) / self.FPS
            if time_taken_seconds < self.WIN_TIME_SECONDS:
                terminal_reward = 100.0 # Victory bonus
                # sfx: win_fast
            else:
                # sfx: win_slow
                pass
        elif self.time_remaining <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            terminal_reward = -100.0 # Timeout penalty
            # sfx: lose
        return terminated, terminal_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)

        # Draw flash effects
        for flash in self.flash_effects:
            fx, fy = flash['pos']
            alpha = int(255 * (flash['timer'] / 10.0))
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, (fx * self.CELL_SIZE, fy * self.CELL_SIZE))

        # Draw numbers
        for num in self.numbers:
            nx, ny = num['pos']
            lerp_factor = (num['val'] - 1) / (self.NUM_TARGETS - 1)
            color = self._lerp_color(self.COLOR_NUMBER_BASE, self.COLOR_NUMBER_MAX, lerp_factor)
            text_surf = self.font_number.render(str(num['val']), True, color)
            text_rect = text_surf.get_rect(center=(
                int(nx * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(ny * self.CELL_SIZE + self.CELL_SIZE / 2)
            ))
            self.screen.blit(text_surf, text_rect)

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE
        ).inflate(-4, -4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=4)


    def _render_ui(self):
        # Eaten count
        eaten_text = f"Eaten: {self.numbers_eaten}/{self.NUM_TARGETS}"
        eaten_surf = self.font_ui.render(eaten_text, True, self.COLOR_TEXT_UI)
        self.screen.blit(eaten_surf, (10, 5))

        # Timer
        time_sec = self.time_remaining / self.FPS
        time_color = self.COLOR_TEXT_UI if time_sec > 5 else self.COLOR_LOSE
        time_text = f"Time: {time_sec:.1f}"
        time_surf = self.font_ui.render(time_text, True, time_color)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(time_surf, time_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.win:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_WIN
            else:
                msg_text = "TIME UP!"
                msg_color = self.COLOR_LOSE
            
            msg_surf = self.font_game_over.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            overlay.blit(msg_surf, msg_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "numbers_eaten": self.numbers_eaten,
            "time_remaining_seconds": self.time_remaining / self.FPS
        }

    def _get_dist_to_target(self):
        target = self._get_target_number_obj()
        if not target:
            return 0
        px, py = self.player_pos
        tx, ty = target['pos']
        return abs(px - tx) + abs(py - ty)

    def _get_target_number_obj(self):
        for num in self.numbers:
            if num['val'] == self.current_target_number:
                return num
        return None

    @staticmethod
    def _lerp_color(c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for manual play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Eater")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    while running:
        # --- Action mapping from keyboard ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Handle Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}")
            print("Press 'R' to play again or close the window.")
        
        # --- Render the observation to the screen ---
        # The observation is (H, W, C), but pygame blits (W, H) surfaces
        # So we need to get the pygame surface directly from the env
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control the frame rate ---
        clock.tick(env.FPS)
        
    env.close()