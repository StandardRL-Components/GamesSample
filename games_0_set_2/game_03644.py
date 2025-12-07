
# Generated: 2025-08-27T23:58:32.076204
# Source Brief: brief_03644.md
# Brief Index: 3644

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the gem. Reach the gold tile before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Navigate the gem to the target, avoiding obstacles. Each move counts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    # Colors
    COLOR_BG = (20, 30, 50)
    COLOR_GRID = (40, 60, 90)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_OBSTACLE_OUTLINE = (60, 60, 70)
    COLOR_TARGET = (255, 215, 0)
    COLOR_GEM = (0, 255, 255)
    COLOR_GEM_GLOW = (150, 255, 255, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (50, 70, 110)
    COLOR_UI_BAR_FG = (100, 150, 255)

    # Grid and Tile
    GRID_WIDTH = 12
    GRID_HEIGHT = 12
    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    
    # Game parameters
    MAX_MOVES = 15
    MAX_OBSTACLES = 10
    INITIAL_OBSTACLES = 3

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
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gem_pos = [0, 0]
        self.target_pos = [0, 0]
        self.obstacle_pos = []
        self.moves_remaining = 0
        self.particles = []
        
        # Difficulty progression
        self.episode_success_count = 0
        self.obstacle_count = self.INITIAL_OBSTACLES
        
        # RNG
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.particles.clear()
        
        # Generate a valid level
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Generates a new level, ensuring a path from start to target exists."""
        generation_attempts = 0
        while True:
            # Place target
            self.target_pos = [
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            ]
            
            # Place gem
            while True:
                self.gem_pos = [
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                ]
                if self.gem_pos != self.target_pos:
                    break
            
            # Place obstacles
            self.obstacle_pos.clear()
            occupied = {tuple(self.gem_pos), tuple(self.target_pos)}
            for _ in range(self.obstacle_count):
                while True:
                    pos = (
                        self.np_random.integers(0, self.GRID_WIDTH),
                        self.np_random.integers(0, self.GRID_HEIGHT)
                    )
                    if pos not in occupied:
                        self.obstacle_pos.append(list(pos))
                        occupied.add(pos)
                        break

            if self._is_path_possible():
                break
            
            generation_attempts += 1
            if generation_attempts > 100:
                # Fallback to a simpler configuration if generation fails
                self.obstacle_count = max(1, self.obstacle_count - 1)
                generation_attempts = 0
    
    def _is_path_possible(self):
        """Checks for a path using Breadth-First Search (BFS)."""
        queue = deque([self.gem_pos])
        visited = {tuple(self.gem_pos)}
        obstacles = {tuple(o) for o in self.obstacle_pos}

        while queue:
            x, y = queue.popleft()

            if [x, y] == self.target_pos:
                return True

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_WIDTH and
                    0 <= ny < self.GRID_HEIGHT and
                    (nx, ny) not in visited and
                    (nx, ny) not in obstacles):
                    
                    visited.add((nx, ny))
                    queue.append([nx, ny])
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        
        old_dist = self._manhattan_distance(self.gem_pos, self.target_pos)
        
        moved = False
        if movement != 0: # 0 is no-op
            self.moves_remaining -= 1
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1  # Right

            next_pos = [self.gem_pos[0] + dx, self.gem_pos[1] + dy]
            
            # Check boundaries and obstacles
            if (0 <= next_pos[0] < self.GRID_WIDTH and
                0 <= next_pos[1] < self.GRID_HEIGHT and
                next_pos not in self.obstacle_pos):
                self.gem_pos = next_pos
                moved = True
                # sfx: gem_move.wav
                self._create_particles(self.gem_pos, self.COLOR_GEM, 20)

        self.steps += 1
        
        new_dist = self._manhattan_distance(self.gem_pos, self.target_pos)
        
        reward = self._calculate_reward(old_dist, new_dist, moved)
        self.score += reward
        
        terminated = self._check_termination()
        
        if terminated and self.gem_pos == self.target_pos:
            self.episode_success_count += 1
            if self.episode_success_count > 0 and self.episode_success_count % 5 == 0:
                self.obstacle_count = min(self.MAX_OBSTACLES, self.obstacle_count + 1)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _calculate_reward(self, old_dist, new_dist, moved):
        reward = 0
        
        # Reward for getting closer, penalize for getting further
        if moved:
            if new_dist < old_dist:
                reward += 0.1
            elif new_dist > old_dist:
                reward -= 0.1
        
        # Large reward for reaching the target
        if self.gem_pos == self.target_pos:
            reward += 5.0 + self.moves_remaining
            # sfx: win.wav
            
        return reward

    def _check_termination(self):
        if self.gem_pos == self.target_pos:
            self.game_over = True
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            # sfx: lose.wav
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "gem_pos": self.gem_pos,
            "target_pos": self.target_pos
        }

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = 320 + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = 100 + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, x, y, color, outline_color=None, surface=None):
        """Draws a single isometric tile (rhombus)."""
        if surface is None:
            surface = self.screen
        
        px, py = self._iso_to_screen(x, y)
        points = [
            (px, py - self.TILE_HEIGHT / 2),
            (px + self.TILE_WIDTH / 2, py),
            (px, py + self.TILE_HEIGHT / 2),
            (px - self.TILE_WIDTH / 2, py),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_tile(x, y, self.COLOR_GRID, self.COLOR_GRID)

        # Draw obstacles
        for ox, oy in self.obstacle_pos:
            self._draw_iso_tile(ox, oy, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_OUTLINE)

        # Draw pulsating target
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
        r, g, b = self.COLOR_TARGET
        pulse_color = (
            int(r * (0.8 + 0.2 * pulse)),
            int(g * (0.8 + 0.2 * pulse)),
            int(b * (0.8 + 0.2 * pulse))
        )
        self._draw_iso_tile(self.target_pos[0], self.target_pos[1], pulse_color)

        # Draw gem
        gem_px, gem_py = self._iso_to_screen(self.gem_pos[0], self.gem_pos[1])
        radius = int(self.TILE_WIDTH / 3)
        
        # Glow effect
        glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_GEM_GLOW, (radius*2, radius*2), radius*2)
        self.screen.blit(glow_surf, (gem_px - radius*2, gem_py - radius*2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_circle(self.screen, gem_px, gem_py, radius, self.COLOR_GEM)
        pygame.gfxdraw.aacircle(self.screen, gem_px, gem_py, radius, self.COLOR_GEM)
        
        # Highlight
        pygame.gfxdraw.filled_circle(self.screen, gem_px - 3, gem_py - 3, 4, (255, 255, 255, 150))

    def _create_particles(self, pos, color, count):
        px, py = self._iso_to_screen(pos[0], pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            
            # Draw particle
            radius = max(0, int(p['life'] / 6))
            if radius > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color']
                )

        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Moves remaining text
        moves_text = self.font_large.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        text_rect = moves_text.get_rect(topright=(620, 20))
        self.screen.blit(moves_text, text_rect)

        # Moves remaining bar
        bar_width = 600
        bar_height = 15
        bar_x = 20
        bar_y = 365
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        if self.moves_remaining > 0:
            fill_width = (self.moves_remaining / self.MAX_MOVES) * bar_width
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Isometric Gem Puzzle")
    clock = pygame.time.Clock()
    
    terminated = False
    total_score = 0
    
    print(env.user_guide)
    
    while not terminated:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
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
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    total_score = 0
                    print("--- Game Reset ---")

        if action[0] != 0: # Only step if a move key was pressed
            obs, reward, term, trunc, info = env.step(action)
            total_score += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Score: {total_score:.2f}, Terminated: {term}")
            if term:
                print("Game Over! Press 'R' to play again.")
                
        # Draw the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()