
# Generated: 2025-08-28T05:30:55.019314
# Source Brief: brief_05608.md
# Brief Index: 5608

        
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
    GameEnv: A grid-based puzzle game where a robot must collect all keys
    and reach the exit within a limited number of moves. The environment
    is designed for visual clarity and a satisfying, turn-based gameplay
    experience, conforming to the Gymnasium API.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the robot one cell at a time. "
        "Collect all the gold keys to unlock the green exit door."
    )

    game_description = (
        "A minimalist puzzle game. Navigate your robot through the grid to collect 5 keys "
        "and reach the exit. You only have 50 moves, so plan your path carefully!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.MAX_MOVES = 50
        self.NUM_KEYS = 5

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_ROBOT_GLOW = (150, 200, 255)
        self.COLOR_KEY = (255, 215, 0)
        self.COLOR_KEY_GLOW = (255, 245, 150)
        self.COLOR_EXIT = (50, 200, 50)
        self.COLOR_EXIT_ACTIVE = (100, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 80, 80)
        self.COLOR_WIN = (100, 255, 100)

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
        self.font = pygame.font.Font(None, 32)
        self.large_font = pygame.font.Font(None, 72)

        # --- Game State (initialized in reset) ---
        self.robot_pos = (0, 0)
        self.key_locations = []
        self.exit_pos = (0, 0)
        self.moves_left = 0
        self.keys_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Visual Effects State
        self.particles = []
        self.move_effect_timer = 0
        self.move_effect_pos = (0, 0)

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        self.keys_collected = 0
        
        # Clear visual effects
        self.particles = []
        self.move_effect_timer = 0

        # Place entities on the grid
        self._place_entities()

        return self._get_observation(), self._get_info()

    def _place_entities(self):
        """Randomly places the robot, keys, and exit on the grid without overlap."""
        all_cells = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        
        # Use the seeded random number generator from Gymnasium
        placements = self.np_random.choice(
            len(all_cells), size=self.NUM_KEYS + 2, replace=False
        )
        
        selected_cells = [all_cells[i] for i in placements]
        
        self.robot_pos = selected_cells.pop()
        self.exit_pos = selected_cells.pop()
        self.key_locations = selected_cells

    def step(self, action):
        movement = action[0]
        reward = 0.0
        terminated = False

        if self.game_over:
            # If the game has already ended, do nothing and return the terminal state.
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Game Logic ---
        self.steps += 1

        # Handle movement action
        if movement != 0:  # 0 is no-op
            self.moves_left -= 1
            old_pos = self.robot_pos
            new_pos = list(old_pos)
            
            dist_before = self._get_dist_to_nearest_target(old_pos)

            if movement == 1: new_pos[1] -= 1  # Up
            elif movement == 2: new_pos[1] += 1  # Down
            elif movement == 3: new_pos[0] -= 1  # Left
            elif movement == 4: new_pos[0] += 1  # Right
            
            # Check grid boundaries
            if 0 <= new_pos[0] < self.GRID_COLS and 0 <= new_pos[1] < self.GRID_ROWS:
                self.robot_pos = tuple(new_pos)
                # # Sound: Robot move sfx
                
                # Trigger visual effect for movement
                self.move_effect_timer = 5  # Effect lasts 5 frames/steps
                self.move_effect_pos = self.robot_pos

                # Calculate distance-based reward
                dist_after = self._get_dist_to_nearest_target(self.robot_pos)
                if dist_after < dist_before:
                    reward += 0.1
                elif dist_after > dist_before:
                    reward -= 0.1
            # else: Robot bumps into wall, no position change.

        # Check for key collection
        if self.robot_pos in self.key_locations:
            self.key_locations.remove(self.robot_pos)
            self.keys_collected += 1
            reward += 10
            self._create_particles(self.robot_pos, self.COLOR_KEY)
            # # Sound: Key collect sfx

        # --- Check Termination Conditions ---
        if self.keys_collected == self.NUM_KEYS and self.robot_pos == self.exit_pos:
            reward += 100
            terminated = True
            self.game_over = True
            self.win = True
            # # Sound: Win jingle
        elif self.moves_left <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
            self.win = False
            # # Sound: Lose buzzer

        # Update visual effects
        if self.move_effect_timer > 0:
            self.move_effect_timer -= 1
        self._update_particles()
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_dist_to_nearest_target(self, pos):
        """Calculates Manhattan distance to the nearest key, or the exit if no keys are left."""
        if not self.key_locations:
            return abs(pos[0] - self.exit_pos[0]) + abs(pos[1] - self.exit_pos[1])
        
        return min(
            abs(pos[0] - k[0]) + abs(pos[1] - k[1]) for k in self.key_locations
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all primary game elements."""
        # Draw grid
        for x in range(self.GRID_COLS + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        # Draw exit
        self._render_exit()

        # Draw keys
        for kx, ky in self.key_locations:
            self._render_key((kx, ky))

        # Draw move effect
        self._render_move_effect()

        # Draw robot
        self._render_robot()

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_exit(self):
        exit_px, exit_py = self.exit_pos[0] * self.CELL_SIZE, self.exit_pos[1] * self.CELL_SIZE
        exit_rect = pygame.Rect(exit_px, exit_py, self.CELL_SIZE, self.CELL_SIZE)
        center = exit_rect.center
        
        if self.keys_collected == self.NUM_KEYS:
            # Pulsing glow effect for active exit
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            glow_size = int(self.CELL_SIZE * (1.2 + pulse * 0.3))
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_EXIT_ACTIVE, 60), (glow_size//2, glow_size//2), glow_size//2)
            self.screen.blit(glow_surf, (center[0] - glow_size//2, center[1] - glow_size//2))
            pygame.draw.rect(self.screen, self.COLOR_EXIT_ACTIVE, exit_rect.inflate(-8, -8), border_radius=4)
        else:
            pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-8, -8), border_radius=4)

    def _render_key(self, pos):
        key_px, key_py = pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE
        center = (key_px + self.CELL_SIZE//2, key_py + self.CELL_SIZE//2)
        
        # Pulsing size effect for keys
        pulse = (math.sin(self.steps * 0.15 + pos[0]) + 1) / 2
        size = int(self.CELL_SIZE * 0.3 + pulse * 4)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], size + 3, (*self.COLOR_KEY_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], size + 3, (*self.COLOR_KEY_GLOW, 100))
        pygame.draw.circle(self.screen, self.COLOR_KEY, center, size)

    def _render_robot(self):
        robot_px = self.robot_pos[0] * self.CELL_SIZE
        robot_py = self.robot_pos[1] * self.CELL_SIZE
        robot_rect = pygame.Rect(robot_px + 5, robot_py + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_GLOW, robot_rect, width=2, border_radius=4)

    def _render_move_effect(self):
        if self.move_effect_timer > 0:
            progress = (5 - self.move_effect_timer) / 5.0
            radius = int(progress * self.CELL_SIZE * 0.7)
            alpha = int((1 - progress) * 150)
            center_px = (self.move_effect_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                         self.move_effect_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            if radius > 0 and alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, (*self.COLOR_ROBOT_GLOW, alpha))
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius, (*self.COLOR_ROBOT_GLOW, alpha))

    def _render_ui(self):
        """Renders text and overlays."""
        # Keys collected
        key_text = self.font.render(f"Keys: {self.keys_collected} / {self.NUM_KEYS}", True, self.COLOR_TEXT)
        self.screen.blit(key_text, (15, 10))

        # Moves left
        moves_text = self.font.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 15, 10))

        # Game Over / Win message
        if self.game_over:
            msg, color = ("YOU WIN!", self.COLOR_WIN) if self.win else ("GAME OVER", self.COLOR_GAMEOVER)
            end_text = self.large_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((20, 20, 30, 200))
            self.screen.blit(s, bg_rect)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "keys_collected": self.keys_collected,
        }

    def _create_particles(self, pos, color):
        center_px = (pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                     pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
        for _ in range(25):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': center_px[0], 'y': center_px[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': random.uniform(2, 5),
                'life': random.randint(20, 40),
                'color': color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 1]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] *= 0.98

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()