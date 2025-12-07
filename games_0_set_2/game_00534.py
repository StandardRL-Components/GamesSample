
# Generated: 2025-08-27T13:56:23.302704
# Source Brief: brief_00534.md
# Brief Index: 534

        
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
    A fast-paced grid-based arcade game where the player collects gems for points
    while racing against a step-based clock.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your avatar on the grid. "
        "Collect gems to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade gem collector. Move around the grid to collect 50 gems "
        "before the time (1000 steps) runs out. Collecting gems gives you points and a reward."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2

        self.MAX_STEPS = 1000
        self.WIN_GEMS = 50
        self.NUM_GEMS_ON_SCREEN = 5
        
        # --- Visuals ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.GEM_COLORS = [(255, 80, 80), (80, 255, 80), (80, 150, 255), (255, 255, 80)]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.player_pos = None
        self.gems = None
        self.steps = None
        self.score = None
        self.collected_gems = None
        self.game_over = None
        self.particles = None
        self.rng = None

        # Initialize state variables
        self.reset()
        
        # Self-check to ensure compliance
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.player_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2], dtype=int)
        
        self.gems = []
        for _ in range(self.NUM_GEMS_ON_SCREEN):
            self._spawn_gem()

        self.steps = 0
        self.score = 0
        self.collected_gems = 0
        self.game_over = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        # space_held and shift_held are unused in this game
        
        self.steps += 1
        reward = 0.0

        # Calculate distance to nearest gem before moving
        dist_before_move = self._get_dist_to_nearest_gem(self.player_pos)

        # Update player position
        prev_pos = self.player_pos.copy()
        if movement == 1: self.player_pos[1] -= 1  # Up
        elif movement == 2: self.player_pos[1] += 1  # Down
        elif movement == 3: self.player_pos[0] -= 1  # Left
        elif movement == 4: self.player_pos[0] += 1  # Right
        
        # Clamp player position to grid bounds
        self.player_pos = np.clip(self.player_pos, 0, self.GRID_SIZE - 1)

        # Create movement particles if the player moved
        if not np.array_equal(prev_pos, self.player_pos):
            self._create_move_particles(prev_pos)

        # Calculate movement-based reward
        dist_after_move = self._get_dist_to_nearest_gem(self.player_pos)
        if dist_after_move < dist_before_move:
            reward += 1.0  # Closer to a gem
        elif movement != 0: # Only penalize if a move was made
             reward -= 0.1 # Further from a gem

        # Check for gem collection
        gem_collected_index = -1
        for i, gem in enumerate(self.gems):
            if np.array_equal(self.player_pos, gem['pos']):
                gem_collected_index = i
                break
        
        if gem_collected_index != -1:
            # Sound: Gem collect
            collected_gem = self.gems.pop(gem_collected_index)
            self._create_gem_particles(collected_gem['pos'], collected_gem['color'])
            
            reward += 10.0
            self.score += 10
            self.collected_gems += 1
            self._spawn_gem()

        # Update game logic
        self._update_particles()
        
        # Check termination conditions
        terminated = False
        if self.collected_gems >= self.WIN_GEMS:
            reward += 100.0  # Win bonus
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 100.0  # Loss penalty
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "collected_gems": self.collected_gems,
        }

    def _grid_to_pixel(self, grid_pos):
        px = self.X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _get_dist_to_nearest_gem(self, pos):
        if not self.gems:
            return float('inf')
        distances = [np.sum(np.abs(pos - gem['pos'])) for gem in self.gems]
        return min(distances) if distances else float('inf')

    def _spawn_gem(self):
        while True:
            new_pos = self.rng.integers(0, self.GRID_SIZE, size=2, dtype=int)
            is_empty = True
            if np.array_equal(new_pos, self.player_pos):
                is_empty = False
            if any(np.array_equal(new_pos, gem['pos']) for gem in self.gems):
                is_empty = False
            
            if is_empty:
                color = self.rng.choice(self.GEM_COLORS)
                self.gems.append({'pos': new_pos, 'color': color})
                break
    
    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_SIZE + 1):
            px = self.X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.Y_OFFSET), (px, self.Y_OFFSET + self.GRID_HEIGHT))
        for y in range(self.GRID_SIZE + 1):
            py = self.Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, py), (self.X_OFFSET + self.GRID_WIDTH, py))

        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            color = p['color']
            alpha = int(255 * life_ratio)
            radius = int(p['radius'] * life_ratio)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*color, alpha))
        
        # Draw gems
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        for gem in self.gems:
            px, py = self._grid_to_pixel(gem['pos'])
            size = int(self.CELL_SIZE * 0.6)
            glow_size = int(size * (1.5 + pulse * 0.5))
            
            glow_color = (*gem['color'], 60)
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_size, glow_color)
            
            r = pygame.Rect(px - size//2, py - size//2, size, size)
            pygame.draw.rect(self.screen, gem['color'], r, border_radius=2)

        # Draw player
        px, py = self._grid_to_pixel(self.player_pos)
        size = int(self.CELL_SIZE * 0.8)
        glow_size = int(size * 1.8)
        
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_size, (*self.COLOR_PLAYER, 30))
        pygame.gfxdraw.filled_circle(self.screen, px, py, int(glow_size * 0.6), (*self.COLOR_PLAYER, 40))
        
        r = pygame.Rect(px - size//2, py - size//2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, r, border_radius=3)
        
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Gems collected
        gems_text = self.font_small.render(f"GEMS: {self.collected_gems} / {self.WIN_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (20, 50))

        # Timer (steps remaining)
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_color = (255, 100, 100) if time_left < 200 else self.COLOR_TEXT
        time_text = self.font_large.render(f"TIME: {time_left}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(time_text, time_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.collected_gems >= self.WIN_GEMS:
                end_text_str = "YOU WIN!"
                end_color = (100, 255, 100)
            else:
                end_text_str = "GAME OVER"
                end_color = (255, 100, 100)
                
            end_text = self.font_large.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _create_gem_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(30):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.integers(20, 40)
            radius = self.rng.integers(3, 7)
            self.particles.append({
                'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'radius': radius, 'color': color
            })

    def _create_move_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(5):
            vel = [(self.rng.random() - 0.5) * 1, (self.rng.random() - 0.5) * 1]
            life = self.rng.integers(10, 20)
            radius = self.rng.integers(1, 3)
            self.particles.append({
                'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'radius': radius, 'color': self.COLOR_PLAYER
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        
        # Test edge cases
        assert self.player_pos[0] >= 0 and self.player_pos[0] < self.GRID_SIZE
        assert self.player_pos[1] >= 0 and self.player_pos[1] < self.GRID_SIZE
        assert self.score >= 0
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Human Controls ---")
    print(env.user_guide)
    print("----------------------")
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        if keys[pygame.K_r]: # Press 'R' to reset
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")

        if keys[pygame.K_ESCAPE]:
            running = False

        # Only step if a movement key is pressed, for turn-based feel
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Gems: {info['collected_gems']}")

            if terminated:
                print("--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Final Reward: {total_reward:.2f}")
                # Wait for a moment before auto-resetting or quitting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the human play speed
        clock.tick(10) # Limit to 10 moves per second

    env.close()