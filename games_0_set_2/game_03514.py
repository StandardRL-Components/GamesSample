
# Generated: 2025-08-27T23:34:27.140114
# Source Brief: brief_03514.md
# Brief Index: 3514

        
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
    """
    An arcade-style game where a ninja collects numbers on a grid before time runs out.
    The environment is designed for visual appeal and satisfying gameplay feel,
    with smooth animations and clear feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the ninja on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide the ninja to collect all 20 numbers before the 30-second timer runs out. "
        "Moving closer to a number gives a small reward, while collecting one gives a large bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

    FPS = 30
    MAX_STEPS = 30 * FPS  # 30 seconds
    NUM_TARGETS = 20

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_NINJA = (0, 150, 255)
    COLOR_NINJA_GLOW = (0, 80, 200)
    COLOR_TARGET = (50, 255, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER = (255, 200, 0)
    COLOR_WIN = (255, 215, 0)
    COLOR_LOSE = (255, 50, 50)

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables - these are reset in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.ninja_pos = [0, 0]
        self.ninja_visual_pos = [0.0, 0.0]
        self.targets = []
        self.particles = []
        self.last_dist_to_target = float('inf')

        self.reset()
        
        # Self-validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Place ninja in the center
        self.ninja_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.ninja_visual_pos = self._grid_to_pixel(self.ninja_pos)

        # Generate targets
        self.targets = []
        occupied_positions = {tuple(self.ninja_pos)}
        while len(self.targets) < self.NUM_TARGETS:
            pos = [
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            ]
            if tuple(pos) not in occupied_positions:
                self.targets.append(pos)
                occupied_positions.add(tuple(pos))
        
        self.particles = []

        # Initialize distance for reward calculation
        _, self.last_dist_to_target = self._find_closest_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, no actions should change the state
            reward = 0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = self._update_game_state(movement)
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.win:
                reward += 100
                self.score += 100
            else: # Timeout
                reward -= 50

        # Update animations
        self._update_visuals()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement):
        reward = 0

        # --- Movement and Distance Reward ---
        if movement > 0: # If not no-op
            next_pos = list(self.ninja_pos)
            if movement == 1:  # Up
                next_pos[1] -= 1
            elif movement == 2:  # Down
                next_pos[1] += 1
            elif movement == 3:  # Left
                next_pos[0] -= 1
            elif movement == 4:  # Right
                next_pos[0] += 1

            # Clamp to grid boundaries
            next_pos[0] = max(0, min(self.GRID_WIDTH - 1, next_pos[0]))
            next_pos[1] = max(0, min(self.GRID_HEIGHT - 1, next_pos[1]))
            
            self.ninja_pos = next_pos
        
        # Calculate reward for moving closer/further
        if self.targets:
            _, new_dist = self._find_closest_target()
            if new_dist < self.last_dist_to_target:
                reward += 1.0
            elif new_dist > self.last_dist_to_target:
                reward -= 0.1
            self.last_dist_to_target = new_dist
        
        # --- Collection ---
        collected_target = None
        for target in self.targets:
            if self.ninja_pos == target:
                collected_target = target
                break
        
        if collected_target:
            self.targets.remove(collected_target)
            reward += 10.0
            self.score += 10
            # SFX: Collect sound
            self._create_particles(self._grid_to_pixel(collected_target), self.COLOR_TARGET)
            # Recalculate distance to new closest target
            if self.targets:
                _, self.last_dist_to_target = self._find_closest_target()
            else:
                self.last_dist_to_target = 0

        return reward

    def _update_visuals(self):
        # Interpolate ninja position for smooth movement
        target_pixel_pos = self._grid_to_pixel(self.ninja_pos)
        self.ninja_visual_pos[0] = self._lerp(self.ninja_visual_pos[0], target_pixel_pos[0], 0.5)
        self.ninja_visual_pos[1] = self._lerp(self.ninja_visual_pos[1], target_pixel_pos[1], 0.5)

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        self.win = len(self.targets) == 0
        time_out = self.steps >= self.MAX_STEPS
        return self.win or time_out

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw targets
        target_radius = min(self.CELL_WIDTH, self.CELL_HEIGHT) // 3
        for target_pos in self.targets:
            px, py = self._grid_to_pixel(target_pos)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), target_radius, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), target_radius, self.COLOR_TARGET)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.draw.circle(self.screen, color, p['pos'], p['radius'] * (p['life'] / p['max_life']))

        # Draw ninja
        ninja_radius = min(self.CELL_WIDTH, self.CELL_HEIGHT) // 2 - 2
        px, py = self.ninja_visual_pos
        
        # Glow effect
        for i in range(ninja_radius // 2, 0, -2):
            alpha = 60 - int(i * (60 / (ninja_radius // 2)))
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), ninja_radius + i, (*self.COLOR_NINJA_GLOW, alpha))
        
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), ninja_radius, self.COLOR_NINJA)
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), ninja_radius, self.COLOR_NINJA)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_color = self.COLOR_TIMER if time_left > 5 else self.COLOR_LOSE
        time_surf = self.font_ui.render(f"Time: {max(0, time_left):.1f}", True, time_color)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_surf, time_rect)

        # Game Over Message
        if self.game_over:
            if self.win:
                msg_surf = self.font_game_over.render("VICTORY!", True, self.COLOR_WIN)
            else:
                msg_surf = self.font_game_over.render("TIME'S UP", True, self.COLOR_LOSE)
            
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Simple background for readability
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "targets_left": len(self.targets),
            "time_left_seconds": (self.MAX_STEPS - self.steps) / self.FPS
        }

    # --- Helper Methods ---
    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        py = grid_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        return [px, py]

    def _find_closest_target(self):
        if not self.targets:
            return None, 0
        
        closest_target = None
        min_dist = float('inf')

        for target in self.targets:
            dist = abs(self.ninja_pos[0] - target[0]) + abs(self.ninja_pos[1] - target[1])
            if dist < min_dist:
                min_dist = dist
                closest_target = target
        
        return closest_target, min_dist
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    @staticmethod
    def _lerp(start, end, t):
        return start + t * (end - start)

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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial info:", info)

    terminated = False
    total_reward = 0
    for i in range(1000):
        if terminated:
            print(f"Game ended at step {i}. Final score: {info['score']}. Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
        
        action = env.action_space.sample() # Replace with agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    print("Simulation finished.")
    env.close()