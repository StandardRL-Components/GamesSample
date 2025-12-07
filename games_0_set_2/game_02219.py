
# Generated: 2025-08-28T04:07:44.866977
# Source Brief: brief_02219.md
# Brief Index: 2219

        
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
        "Controls: Use arrow keys to move the cursor. Press space to clear a highlighted gem cluster (3+ gems). "
        "Each action costs one move. Press Shift on the game over screen to restart."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic gem-matching puzzle. Clear clusters of 3 or more gems to score points. "
        "Reach 100 points within 20 moves to win. Larger clusters give bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.screen_width = 640
        self.screen_height = 400
        self.grid_cols, self.grid_rows = 12, 8
        self.gem_size = 40
        self.grid_width = self.grid_cols * self.gem_size
        self.grid_height = self.grid_rows * self.gem_size
        self.grid_offset_x = (self.screen_width - self.grid_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_height) // 2

        # Game constants
        self.win_score = 100
        self.start_moves = 20
        self.min_cluster_size = 3
        self.large_cluster_bonus_threshold = 4
        self.large_cluster_bonus = 5
        self.win_bonus = 50

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_title = pygame.font.SysFont("Arial", 48, bold=True)

        # Colors and Gem Info
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.GEM_TYPES = {
            1: {"color": (220, 50, 50), "value": 1},  # Red
            2: {"color": (50, 220, 50), "value": 2},  # Green
            3: {"color": (80, 80, 255), "value": 3},  # Blue
            4: {"color": (255, 220, 50), "value": 4}, # Yellow
        }

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.last_reward = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.moves_remaining = self.start_moves
        self.game_over = False
        self.win = False
        self.cursor_pos = [self.grid_cols // 2, self.grid_rows // 2]
        self.particles = []
        self.last_reward = 0
        
        self._create_and_validate_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        if self.game_over:
            if shift_held:
                # Allow reset only on game over screen
                obs, info = self.reset()
                return obs, 0, False, False, info
            else:
                # No change if game is over and not resetting
                return self._get_observation(), 0, True, False, self._get_info()

        self.moves_remaining -= 1

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.grid_rows - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.grid_cols - 1, self.cursor_pos[0] + 1)

        # 2. Handle gem clearing action
        if space_held:
            cluster = self._find_cluster(self.cursor_pos[0], self.cursor_pos[1])
            if len(cluster) >= self.min_cluster_size:
                # SFX: Gem Clear Success
                
                # Calculate score from gems
                cleared_value = 0
                for x, y in cluster:
                    gem_type = self.grid[y][x]
                    if gem_type > 0:
                        cleared_value += self.GEM_TYPES[gem_type]["value"]
                
                reward += cleared_value
                self.score += cleared_value

                # Bonus for large clusters
                if len(cluster) > self.large_cluster_bonus_threshold:
                    reward += self.large_cluster_bonus
                    self.score += self.large_cluster_bonus

                # Process grid changes
                self._create_particles(cluster)
                self._remove_gems(cluster)
                self._apply_gravity_and_refill()

                # Anti-softlock: if no moves left, reshuffle
                if not self._check_for_any_valid_moves():
                    # SFX: Grid Reshuffle
                    self._create_and_validate_grid()
            else:
                # SFX: Action Failed/Invalid
                reward = 0 # No penalty, but move is consumed
        
        self.last_reward = reward

        # 3. Check for termination
        self.win = self.score >= self.win_score
        terminated = self.win or self.moves_remaining <= 0
        self.game_over = terminated
        
        if self.win and terminated: # Grant win bonus only on the terminating step
            reward += self.win_bonus

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.start_moves - self.moves_remaining}

    def _create_and_validate_grid(self):
        while True:
            self.grid = [[self.np_random.integers(1, len(self.GEM_TYPES) + 1) for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
            if self._check_for_any_valid_moves():
                break

    def _check_for_any_valid_moves(self):
        for y in range(self.grid_rows):
            for x in range(self.grid_cols):
                if self.grid[y][x] > 0:
                    if len(self._find_cluster(x, y)) >= self.min_cluster_size:
                        return True
        return False

    def _find_cluster(self, start_x, start_y):
        if not (0 <= start_x < self.grid_cols and 0 <= start_y < self.grid_rows):
            return []
        
        target_type = self.grid[start_y][start_x]
        if target_type == 0:
            return []

        q = [(start_x, start_y)]
        visited = set(q)
        cluster = []

        while q:
            x, y = q.pop(0)
            cluster.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows and (nx, ny) not in visited:
                    if self.grid[ny][nx] == target_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return cluster

    def _remove_gems(self, cluster):
        for x, y in cluster:
            self.grid[y][x] = 0 # Mark as empty

    def _apply_gravity_and_refill(self):
        for x in range(self.grid_cols):
            empty_slots = 0
            for y in range(self.grid_rows - 1, -1, -1):
                if self.grid[y][x] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[y + empty_slots][x] = self.grid[y][x]
                    self.grid[y][x] = 0
            # Refill top
            for y in range(empty_slots):
                self.grid[y][x] = self.np_random.integers(1, len(self.GEM_TYPES) + 1)

    def _create_particles(self, cluster):
        for x, y in cluster:
            gem_type = self.grid[y][x]
            if gem_type == 0: continue
            
            px = self.grid_offset_x + x * self.gem_size + self.gem_size // 2
            py = self.grid_offset_y + y * self.gem_size + self.gem_size // 2
            color = self.GEM_TYPES[gem_type]["color"]

            for _ in range(10): # 10 particles per gem
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                lifetime = self.np_random.integers(15, 30)
                self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 1 # lifetime
            
            radius = int(max(0, (p[4] / 30) * 5))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, p[5])
        
        self.particles = [p for p in self.particles if p[4] > 0]


    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_width, self.grid_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Find highlighted cluster for visual feedback
        highlight_cluster = self._find_cluster(self.cursor_pos[0], self.cursor_pos[1])
        is_valid_cluster = len(highlight_cluster) >= self.min_cluster_size

        # Draw gems and highlights
        for y in range(self.grid_rows):
            for x in range(self.grid_cols):
                gem_type = self.grid[y][x]
                if gem_type > 0:
                    rect = pygame.Rect(
                        self.grid_offset_x + x * self.gem_size,
                        self.grid_offset_y + y * self.gem_size,
                        self.gem_size, self.gem_size
                    )
                    color = self.GEM_TYPES[gem_type]["color"]
                    
                    # Draw gem with a darker border for depth
                    pygame.draw.rect(self.screen, color, rect, border_radius=5)
                    border_color = tuple(max(0, c-40) for c in color)
                    pygame.draw.rect(self.screen, border_color, rect, width=3, border_radius=5)

                    # Highlight effect for valid clusters
                    if is_valid_cluster and (x, y) in highlight_cluster:
                        highlight_surf = pygame.Surface((self.gem_size, self.gem_size), pygame.SRCALPHA)
                        highlight_surf.fill((255, 255, 255, 100))
                        self.screen.blit(highlight_surf, rect.topleft)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.gem_size,
            self.grid_offset_y + self.cursor_pos[1] * self.gem_size,
            self.gem_size, self.gem_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=4, border_radius=5)
        
        # Draw cluster size next to cursor
        if is_valid_cluster:
            cluster_text = self.font_small.render(f"{len(highlight_cluster)}", True, self.COLOR_TEXT)
            text_rect = cluster_text.get_rect(center=(cursor_rect.centerx, cursor_rect.centery))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(8,4), border_radius=5)
            self.screen.blit(cluster_text, text_rect)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves remaining display
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.screen_width - 20, 10))
        self.screen.blit(moves_text, moves_rect)
        
        # Game Over / Win screen
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 220))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                end_text = self.font_title.render("YOU WIN!", True, self.GEM_TYPES[4]["color"])
            else:
                end_text = self.font_title.render("GAME OVER", True, self.GEM_TYPES[1]["color"])
            
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 40))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)
            
            restart_text = self.font_small.render("Press SHIFT to Restart", True, self.COLOR_TEXT)
            restart_rect = restart_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 60))
            self.screen.blit(restart_text, restart_rect)

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
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Gem Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_held = 1
            if event.type == pygame.KEYDOWN and terminated:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1

        # Only step if an action is taken
        if movement != 0 or space_held != 0 or shift_held != 0:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, _, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Terminated: {terminated}")
        else: # If no key is pressed, just re-render the current state
             obs = env._get_observation()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate for human play

    env.close()