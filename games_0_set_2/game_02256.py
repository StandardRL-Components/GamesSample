
# Generated: 2025-08-27T19:46:38.827568
# Source Brief: brief_02256.md
# Brief Index: 2256

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character one tile at a time on the isometric grid."
    )

    game_description = (
        "Navigate an isometric grid, collecting all 10 gems while dodging deadly traps. Each move is a strategic choice in this turn-based puzzle game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_GEMS = 10
        self.NUM_TRAPS = 5
        self.MAX_STEPS = 1000

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 55, 70)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_GLOW = (60, 160, 255, 50)
        self.COLOR_GEM = (255, 220, 50)
        self.COLOR_GEM_GLOW = (255, 220, 50, 70)
        self.COLOR_TRAP = (255, 80, 80)
        self.COLOR_TRAP_GLOW = (255, 80, 80, 80)
        self.COLOR_UI = (220, 220, 220)

        # Isometric projection constants
        self.TILE_WIDTH = 40
        self.TILE_HEIGHT = 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # Initialize state variables
        self.rng = None
        self.player_pos = (0, 0)
        self.gems = set()
        self.traps = set()
        self.particles = []
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False

        self.reset()
        
        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        """Converts grid coordinates to screen coordinates."""
        screen_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _find_closest_dist(self, pos, items):
        """Calculates Manhattan distance to the closest item in a set."""
        if not items:
            return float('inf')
        return min(abs(pos[0] - item[0]) + abs(pos[1] - item[1]) for item in items)

    def _path_exists_to_any_gem(self):
        """Uses BFS to check if there is a walkable path from player to any gem."""
        if not self.gems:
            return True # No gems left to find
            
        q = deque([self.player_pos])
        visited = {self.player_pos}

        while q:
            x, y = q.popleft()

            if (x, y) in self.gems:
                return True

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    neighbor = (nx, ny)
                    if neighbor not in visited and neighbor not in self.traps:
                        visited.add(neighbor)
                        q.append(neighbor)
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.particles.clear()

        # Generate a valid level layout
        while True:
            all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            self.rng.shuffle(all_coords)
            
            # Ensure all_coords has enough elements
            if len(all_coords) < 1 + self.NUM_TRAPS + self.NUM_GEMS:
                continue

            self.player_pos = all_coords.pop(0)
            self.traps = {all_coords.pop(i) for i in range(self.NUM_TRAPS)}
            self.gems = {all_coords.pop(i) for i in range(self.NUM_GEMS)}

            if self._path_exists_to_any_gem():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        terminated = False
        reward = 0.0
        
        old_pos = self.player_pos
        old_dist_gem = self._find_closest_dist(old_pos, self.gems)
        old_dist_trap = self._find_closest_dist(old_pos, self.traps)

        # --- Update Player Position ---
        px, py = self.player_pos
        if movement == 1:  # Up
            py -= 1
        elif movement == 2:  # Down
            py += 1
        elif movement == 3:  # Left
            px -= 1
        elif movement == 4:  # Right
            px += 1

        # Clamp to grid boundaries
        px = max(0, min(self.GRID_SIZE - 1, px))
        py = max(0, min(self.GRID_SIZE - 1, py))
        self.player_pos = (px, py)
        
        # --- Calculate Continuous Reward ---
        if self.player_pos != old_pos:
            new_dist_gem = self._find_closest_dist(self.player_pos, self.gems)
            if new_dist_gem < old_dist_gem:
                reward += 1.0

            new_dist_trap = self._find_closest_dist(self.player_pos, self.traps)
            if new_dist_trap < old_dist_trap:
                reward -= 0.1

        # --- Check for Events ---
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.score += 10
            self.gems_collected += 1
            reward += 10.0
            # sfx: gem collect sound
            self._create_particles(self.player_pos, self.COLOR_GEM, 20)

            if not self.gems: # Victory condition
                reward += 50.0
                terminated = True
                self.game_over = True
                # sfx: victory fanfare

        if self.player_pos in self.traps:
            reward -= 50.0
            terminated = True
            self.game_over = True
            # sfx: player death sound
            self._create_particles(self.player_pos, self.COLOR_TRAP, 40)
        
        # --- Update Game State ---
        self.steps += 1
        self._update_particles()
        
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "gems_collected": self.gems_collected,
            "player_pos": self.player_pos,
        }

    def _render_game(self):
        # Pulsing effect for interactive elements
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 * 3

        # Draw grid cells, traps, and gems
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                screen_pos = self._iso_to_screen(x, y)
                is_trap = (x, y) in self.traps
                is_gem = (x, y) in self.gems

                if is_trap:
                    self._draw_iso_tile(screen_pos, self.COLOR_TRAP, self.COLOR_TRAP_GLOW, int(pulse))
                elif is_gem:
                    self._draw_iso_tile(screen_pos, self.COLOR_GEM, self.COLOR_GEM_GLOW, int(pulse))
                else:
                    self._draw_iso_tile(screen_pos, self.COLOR_GRID)

        # Draw player
        player_screen_pos = self._iso_to_screen(*self.player_pos)
        self._draw_iso_tile(player_screen_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 5)

        # Draw particles
        for p in self.particles:
            p_pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['life'] / p['max_life'] * 4)
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], p_pos, radius)

    def _draw_iso_tile(self, screen_pos, color, glow_color=None, glow_size=0):
        sx, sy = screen_pos
        points = [
            (sx, sy - self.TILE_HEIGHT // 2),
            (sx + self.TILE_WIDTH // 2, sy),
            (sx, sy + self.TILE_HEIGHT // 2),
            (sx - self.TILE_WIDTH // 2, sy),
        ]
        
        if glow_color:
            glow_points = [
                (sx, sy - self.TILE_HEIGHT // 2 - glow_size),
                (sx + self.TILE_WIDTH // 2 + glow_size, sy),
                (sx, sy + self.TILE_HEIGHT // 2 + glow_size),
                (sx - self.TILE_WIDTH // 2 - glow_size, sy),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, glow_color)

        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (15, 10))
        
        gems_text = self.font_small.render(f"Gems: {self.gems_collected} / {self.NUM_GEMS}", True, self.COLOR_UI)
        self.screen.blit(gems_text, (15, 45))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if not self.gems else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_GEM if not self.gems else self.COLOR_TRAP)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, grid_pos, color, count):
        screen_pos = self._iso_to_screen(*grid_pos)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(screen_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Gem Collector")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    while not terminated:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    continue

        # Only step if a movement key was pressed
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated and env.game_over:
            # Wait for a moment to show the game over screen
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    pygame.quit()