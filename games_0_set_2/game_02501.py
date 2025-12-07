
# Generated: 2025-08-28T05:05:23.524045
# Source Brief: brief_02501.md
# Brief Index: 2501

        
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
        "Controls: Use arrow keys to move the cursor. Press space to clear a block cluster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Clear matching block clusters to score points. You have a limited number of moves. Plan ahead to clear the entire board!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 12
    GRID_HEIGHT = 8
    BLOCK_SIZE = 40
    GRID_LINE_WIDTH = 2
    
    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_GRID_LINES = (30, 45, 65)
    COLOR_CURSOR = (255, 255, 255)
    
    # Block type 0 is empty, -1 is unbreakable
    BLOCK_COLORS = {
        -1: ((80, 80, 90), (60, 60, 70)),  # Grey (unbreakable)
        1: ((255, 80, 80), (200, 50, 50)),   # Red
        2: ((80, 255, 80), (50, 200, 50)),   # Green
        3: ((80, 150, 255), (50, 100, 200)),  # Blue
        4: ((255, 255, 80), (200, 200, 50)),  # Yellow
        5: ((255, 80, 255), (200, 50, 200)),  # Magenta
    }
    NUM_COLORS = len(BLOCK_COLORS) - 1

    # Game parameters
    INITIAL_MOVES = 25
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid_pixel_width = self.GRID_WIDTH * self.BLOCK_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.cursor_pos = [0, 0]
        self.particles = []
        self.steps = 0
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.win_message = ""

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.win_message = ""
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.particles = []
        
        self._generate_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # Every action costs one move
        self.moves_remaining -= 1
        
        # 1. Handle cursor movement
        if movement == 1: # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: # Down
            self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
        elif movement == 3: # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: # Right
            self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)

        # 2. Handle block clearing
        if space_pressed:
            r, c = self.cursor_pos
            block_type = self.grid[r, c]

            if block_type > 0: # Is a clearable block
                cluster = self._find_cluster(r, c)
                cluster_size = len(cluster)

                if cluster_size == 1:
                    reward += -0.1 # Penalty for clearing single block
                elif cluster_size > 1:
                    # Clear blocks and get reward
                    cleared_count = 0
                    for br, bc in cluster:
                        self._create_particles(br, bc, self.grid[br, bc])
                        self.grid[br, bc] = 0
                        cleared_count += 1
                    
                    # Reward calculation
                    reward += cleared_count # +1 for each block
                    if cleared_count > 5:
                        reward += 5 # Bonus for large clusters
                    self.score += int(reward) # Update score with integer part of reward
                    
                    # Apply gravity
                    self._apply_gravity()
        
        # 3. Update game state
        self._update_particles()
        self.steps += 1
        
        # 4. Check for termination
        terminated, win = self._check_termination()
        if terminated:
            self.game_over = True
            if win:
                reward += 100
                self.score += 100
                self.win_message = "BOARD CLEARED!"
            else:
                reward += -50
                self.score += -50
                if self.moves_remaining <= 0:
                    self.win_message = "OUT OF MOVES!"
                else:
                    self.win_message = "NO MOVES LEFT!"

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME LIMIT REACHED"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_grid(self):
        # Ensure a playable grid is generated
        while True:
            # Fill with random colored blocks
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            
            # Add some unbreakable blocks
            num_grey = self.np_random.integers(3, 8)
            for _ in range(num_grey):
                r, c = self.np_random.integers(0, self.GRID_HEIGHT), self.np_random.integers(0, self.GRID_WIDTH)
                self.grid[r, c] = -1
            
            if self._check_for_any_valid_move():
                break

    def _find_cluster(self, start_r, start_c):
        target_type = self.grid[start_r, start_c]
        if target_type <= 0:
            return []

        q = deque([(start_r, start_c)])
        visited = set([(start_r, start_c)])
        cluster = []

        while q:
            r, c = q.popleft()
            cluster.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_type:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return cluster
    
    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            write_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != write_row:
                        self.grid[write_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    write_row -= 1

    def _check_for_any_valid_move(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if not visited[r, c] and self.grid[r, c] > 0:
                    cluster = self._find_cluster(r, c)
                    if len(cluster) > 1:
                        return True
                    for cr, cc in cluster:
                        visited[cr, cc] = True
        return False

    def _check_termination(self):
        # Win condition: no colored blocks left
        has_colored_blocks = np.any(self.grid > 0)
        if not has_colored_blocks:
            return True, True # terminated, win
        
        # Loss condition: no moves remaining
        if self.moves_remaining <= 0:
            return True, False # terminated, loss
        
        # Loss condition: no valid moves left
        if not self._check_for_any_valid_move():
            return True, False # terminated, loss
            
        return False, False # not terminated

    def _create_particles(self, r, c, block_type):
        color, _ = self.BLOCK_COLORS[block_type]
        center_x = self.grid_offset_x + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.grid_offset_y + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        for _ in range(15): # Create 15 particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append([center_x, center_y, vel_x, vel_y, lifetime, color, size])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 1    # lifetime -= 1

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
            "moves_remaining": self.moves_remaining,
            "cursor_pos": self.cursor_pos,
        }

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_pixel_width, y), self.GRID_LINE_WIDTH)
        for c in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_pixel_height), self.GRID_LINE_WIDTH)

        # Draw blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                block_type = self.grid[r, c]
                if block_type != 0:
                    self._draw_block(r, c, block_type)

        # Draw particles
        for p in self.particles:
            x, y, _, _, lifetime, color, size = p
            alpha = max(0, min(255, int(255 * (lifetime / 30))))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, (*color, alpha), (0, 0, size, size))
            self.screen.blit(temp_surf, (int(x - size/2), int(y - size/2)), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.BLOCK_SIZE,
            self.grid_offset_y + cursor_r * self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            self.BLOCK_SIZE
        )
        alpha = 100 + 50 * math.sin(pygame.time.get_ticks() * 0.01)
        pygame.gfxdraw.box(self.screen, rect, (*self.COLOR_CURSOR, alpha))

    def _draw_block(self, r, c, block_type):
        main_color, border_color = self.BLOCK_COLORS[block_type]
        margin = 3
        shadow_depth = 3
        
        base_rect = pygame.Rect(
            self.grid_offset_x + c * self.BLOCK_SIZE + margin,
            self.grid_offset_y + r * self.BLOCK_SIZE + margin,
            self.BLOCK_SIZE - 2 * margin,
            self.BLOCK_SIZE - 2 * margin
        )
        
        shadow_rect = base_rect.copy()
        shadow_rect.move_ip(0, shadow_depth)
        
        pygame.draw.rect(self.screen, border_color, shadow_rect, border_radius=4)
        pygame.draw.rect(self.screen, main_color, base_rect, border_radius=4)

    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, (200, 220, 255))
        self.screen.blit(score_text, (20, 10))

        # Moves display
        moves_text = self.font_medium.render(f"MOVES: {self.moves_remaining}", True, (200, 220, 255))
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Breaker Gym Env")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                
                # Take a step if an action was taken
                if any(a != 0 for a in action):
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}, Terminated: {terminated}")
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING GAME ---")
                obs, info = env.reset()
                terminated = False

        # Update display
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # Transpose back from (H, W, C) to (W, H, C) for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()