
# Generated: 2025-08-28T00:47:18.004748
# Source Brief: brief_03904.md
# Brief Index: 3904

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Reach the green exit, avoid red traps."
    )

    game_description = (
        "Navigate a treacherous grid, avoiding traps to reach the exit in the fewest steps possible."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 7
        self.MAX_STEPS = 1000
        self.WIN_STEPS = 30
        self.NUM_TRAPS_MIN = 10
        self.NUM_TRAPS_MAX = 15

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_EXIT_INNER = (150, 255, 200)
        self.COLOR_TRAP = (255, 80, 80)
        self.COLOR_TRAP_INNER = (255, 150, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 15, 20)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Grid layout calculation
        self.cell_size = int(min(self.WIDTH, self.HEIGHT) * 0.8 / self.GRID_SIZE)
        self.grid_width = self.grid_height = self.GRID_SIZE * self.cell_size
        self.offset_x = (self.WIDTH - self.grid_width) // 2
        self.offset_y = (self.HEIGHT - self.grid_height) // 2

        # Initialize state variables
        self.player_pos = None
        self.exit_pos = None
        self.trap_positions = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_message = None
        self.rng = None

        # First reset to initialize state, then validate
        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        """Generates a new level layout ensuring a solvable path."""
        start_pos = [self.GRID_SIZE // 2, self.GRID_SIZE - 1]
        exit_pos = [self.GRID_SIZE // 2, 0]

        # Use BFS to find all reachable cells from the exit
        q = collections.deque([exit_pos])
        visited = {tuple(exit_pos)}
        
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append([nx, ny])

        # Potential trap locations are all reachable cells except start and exit
        potential_trap_cells = list(visited - {tuple(start_pos), tuple(exit_pos)})
        
        num_traps = self.rng.integers(self.NUM_TRAPS_MIN, self.NUM_TRAPS_MAX + 1)
        num_traps = min(num_traps, len(potential_trap_cells))
        
        trap_indices = self.rng.choice(len(potential_trap_cells), size=num_traps, replace=False)
        
        self.player_pos = start_pos
        self.exit_pos = exit_pos
        self.trap_positions = {potential_trap_cells[i] for i in trap_indices}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.2  # Cost of living/taking a step
        
        old_pos = list(self.player_pos)
        old_dist_to_exit = abs(old_pos[0] - self.exit_pos[0]) + abs(old_pos[1] - self.exit_pos[1])

        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if movement != 0:
            self.steps += 1
            
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy

            # Boundary check
            if 0 <= new_x < self.GRID_SIZE and 0 <= new_y < self.GRID_SIZE:
                self.player_pos = [new_x, new_y]
                # sfx: player_move.wav
            else:
                # sfx: wall_bump.wav
                pass # No movement if hitting a wall

        new_dist_to_exit = abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])

        if new_dist_to_exit < old_dist_to_exit:
            reward += 0.5

        terminated = self._check_termination()
        
        if terminated:
            if tuple(self.player_pos) == tuple(self.exit_pos):
                if self.steps <= self.WIN_STEPS:
                    reward = 50.0
                    self.win_message = "YOU WIN!"
                    # sfx: win_fast.wav
                else:
                    reward = 20.0
                    self.win_message = "SUCCESS!"
                    # sfx: win_slow.wav
            elif tuple(self.player_pos) in self.trap_positions:
                reward = -50.0
                self.win_message = "GAME OVER"
                # sfx: lose_trap.wav
            elif self.steps >= self.MAX_STEPS:
                self.win_message = "TIME UP"
                # sfx: lose_timeout.wav

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if tuple(self.player_pos) == tuple(self.exit_pos):
            self.game_over = True
            return True
        if tuple(self.player_pos) in self.trap_positions:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixels(self, grid_pos):
        x = self.offset_x + grid_pos[0] * self.cell_size
        y = self.offset_y + grid_pos[1] * self.cell_size
        return x, y

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.offset_x + i * self.cell_size, self.offset_y), 
                             (self.offset_x + i * self.cell_size, self.offset_y + self.grid_height))
            # Horizontal
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.offset_x, self.offset_y + i * self.cell_size), 
                             (self.offset_x + self.grid_width, self.offset_y + i * self.cell_size))

        # Draw exit
        ex, ey = self._grid_to_pixels(self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex, ey, self.cell_size, self.cell_size))
        inner_pad = self.cell_size // 6
        pygame.draw.rect(self.screen, self.COLOR_EXIT_INNER, 
                         (ex + inner_pad, ey + inner_pad, 
                          self.cell_size - 2 * inner_pad, self.cell_size - 2 * inner_pad), 
                         border_radius=inner_pad // 2)

        # Draw traps
        for trap_pos in self.trap_positions:
            tx, ty = self._grid_to_pixels(trap_pos)
            pygame.draw.rect(self.screen, self.COLOR_TRAP, (tx, ty, self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, self.COLOR_TRAP_INNER, 
                             (tx + inner_pad, ty + inner_pad, 
                              self.cell_size - 2 * inner_pad, self.cell_size - 2 * inner_pad), 
                             border_radius=inner_pad // 2)

        # Draw player
        px, py = self._grid_to_pixels(self.player_pos)
        center_x = px + self.cell_size // 2
        center_y = py + self.cell_size // 2
        radius = self.cell_size // 2 - 4
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER_GLOW)
        
        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius - 2, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius - 2, self.COLOR_PLAYER)

    def _render_text(self, text, font, color, center_pos):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        self.screen.blit(text_surf, text_rect)

    def _render_text_with_shadow(self, text, font, color, shadow_color, pos, anchor="topleft"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect(**{anchor: pos})
        shadow_rect = shadow_surf.get_rect(**{anchor: (pos[0] + 2, pos[1] + 2)})

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Display steps and score
        self._render_text_with_shadow(f"Steps: {self.steps}/{self.MAX_STEPS}", self.font_medium, 
                                      self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, (10, 10))
        self._render_text_with_shadow(f"Score: {self.score:.1f}", self.font_medium, 
                                      self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, (self.WIDTH - 10, 10), anchor="topright")
        
        # Display game over message
        if self.game_over and self.win_message:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            self._render_text_with_shadow(self.win_message, self.font_large, 
                                          self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, 
                                          (self.WIDTH // 2, self.HEIGHT // 2), anchor="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

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
        
        # Test reward range
        assert -50.2 <= reward <= 50.5, f"Reward {reward} out of range"

        # Test state guarantees
        assert 0 <= self.player_pos[0] < self.GRID_SIZE and 0 <= self.player_pos[1] < self.GRID_SIZE
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display drivers
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Navigator")
    clock = pygame.time.Clock()
    
    total_score = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    print("="*30 + "\n")
    
    while not done:
        # Default action is no-op
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                    total_score = 0
                elif event.key == pygame.K_q:
                    done = True

        if done:
            break

        # Only step if a move key was pressed
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_score += reward
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Score: {total_score:.2f}")
            
            if terminated:
                print(f"--- Episode Finished in {info['steps']} steps. Final Score: {total_score:.2f} ---")
                # Wait for a moment before auto-resetting
                # Render final frame
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(3000)
                
                print("Resetting for new episode...")
                obs, info = env.reset()
                total_score = 0

        # Rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()
    print("Game window closed.")