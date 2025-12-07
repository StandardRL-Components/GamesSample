import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to clear a cluster of matching gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by selecting clusters of two or more matching gems. Larger clusters grant bonus points. The game ends when the board is cleared or no more moves are possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 10
        self.NUM_GEM_TYPES = 5
        self.GEM_SIZE = 36
        self.GRID_LINE_WIDTH = 2
        self.MAX_STEPS = 1000

        self.WIDTH = 640
        self.HEIGHT = 400

        total_grid_width = self.GRID_WIDTH * self.GEM_SIZE + (self.GRID_WIDTH - 1) * self.GRID_LINE_WIDTH
        total_grid_height = self.GRID_HEIGHT * self.GEM_SIZE + (self.GRID_HEIGHT - 1) * self.GRID_LINE_WIDTH
        self.BOARD_OFFSET_X = (self.WIDTH - total_grid_width) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - total_grid_height) // 2
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 18)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 18)

        # State variables (initialized in reset)
        self.board = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.valid_moves_count = 0
        
        # This is called at the end of __init__ in the original code,
        # but it's better to reset the state first.
        # self.reset()
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        
        # Generate a board with at least 5 valid moves
        while True:
            self._generate_board()
            self.valid_moves_count = self._count_valid_moves()
            if self.valid_moves_count >= 5:
                break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = 0
        terminated = False
        
        self._handle_movement(movement)
        
        action_taken = False
        if space_held:
            action_taken = True
            reward += self._execute_selection()
        
        if not action_taken and movement == 0:
            reward -= 0.1 # Small penalty for doing nothing

        self.steps += 1
        self._update_particles()
        
        # Check termination conditions
        self.valid_moves_count = self._count_valid_moves()
        
        if np.sum(self.board) == 0: # Board is empty (win)
            reward += 100
            terminated = True
            self.game_over = True
        elif self.valid_moves_count == 0: # No more valid moves (loss)
            reward -= 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS: # Step limit reached
            terminated = True
            self.game_over = True

        truncated = False # This environment does not truncate based on time limits in the same way as other envs

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

    def _execute_selection(self):
        x, y = self.cursor_pos
        if self.board[y][x] == 0: # Empty space
            return -1.0

        cluster = self._find_cluster(x, y)
        
        if len(cluster) > 1:
            # Sound effect placeholder: # sfx_gem_clear.play()
            num_cleared = len(cluster)
            
            # Quadratic reward for larger clusters
            reward = (num_cleared - 1) ** 1.5
            if num_cleared > 5:
                reward += 5 # Bonus for large clusters

            for pos_x, pos_y in cluster:
                gem_type = self.board[pos_y][pos_x]
                self._create_particles(pos_x, pos_y, gem_type)
                self.board[pos_y][pos_x] = 0 # Mark as empty
            
            self.score += int(reward)
            self._apply_gravity_and_refill()
            return reward
        else:
            # Sound effect placeholder: # sfx_invalid_move.play()
            return -1.0 # Penalty for invalid move

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
            "valid_moves": self.valid_moves_count,
            "cursor_pos": self.cursor_pos,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.BOARD_OFFSET_X + i * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH / 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_OFFSET_Y), (x, self.BOARD_OFFSET_Y + self.GRID_HEIGHT * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.BOARD_OFFSET_Y + i * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH / 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_OFFSET_X, y), (self.BOARD_OFFSET_X + self.GRID_WIDTH * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH, y), self.GRID_LINE_WIDTH)

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.board[y][x]
                if gem_type > 0:
                    self._draw_gem(x, y, gem_type)
        
        self._draw_particles()
        self._draw_cursor()

    def _draw_gem(self, x, y, gem_type):
        color_index = gem_type - 1
        base_color = self.GEM_COLORS[color_index]
        highlight_color = tuple(min(255, c + 60) for c in base_color)
        
        px = self.BOARD_OFFSET_X + x * (self.GEM_SIZE + self.GRID_LINE_WIDTH)
        py = self.BOARD_OFFSET_Y + y * (self.GEM_SIZE + self.GRID_LINE_WIDTH)
        
        gem_rect = pygame.Rect(px, py, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, base_color, gem_rect, border_radius=8)
        
        highlight_rect = pygame.Rect(px + 4, py + 4, self.GEM_SIZE - 16, self.GEM_SIZE - 16)
        pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=6)

    def _draw_cursor(self):
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        alpha = 100 + pulse * 100
        
        x, y = self.cursor_pos
        px = self.BOARD_OFFSET_X + x * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - 2
        py = self.BOARD_OFFSET_Y + y * (self.GEM_SIZE + self.GRID_LINE_WIDTH) - 2
        
        cursor_surface = pygame.Surface((self.GEM_SIZE + 4, self.GEM_SIZE + 4), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), cursor_surface.get_rect(), width=4, border_radius=10)
        self.screen.blit(cursor_surface, (px, py))

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        moves_text = self.font_large.render(f"Moves: {self.valid_moves_count}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 15))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "BOARD CLEARED!" if np.sum(self.board) == 0 else "NO MOVES LEFT"
            text_surf = self.font_large.render(status_text, True, (255, 255, 100))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)


    def _generate_board(self):
        self.board = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _find_cluster(self, start_x, start_y):
        target_color = self.board[start_y][start_x]
        if target_color == 0:
            return []
        
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        cluster = []

        while q:
            x, y = q.popleft()
            cluster.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.board[ny][nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return cluster

    def _count_valid_moves(self):
        visited = np.zeros_like(self.board, dtype=bool)
        move_count = 0
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if not visited[y][x] and self.board[y][x] != 0:
                    cluster = self._find_cluster(x, y)
                    if len(cluster) > 1:
                        move_count += 1
                    for cx, cy in cluster:
                        visited[cy][cx] = True
        return move_count

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[y][x] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.board[y + empty_slots][x] = self.board[y][x]
                    self.board[y][x] = 0
            
            # Refill from top
            for y in range(empty_slots):
                self.board[y][x] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _create_particles(self, grid_x, grid_y, gem_type):
        px = self.BOARD_OFFSET_X + grid_x * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE / 2
        py = self.BOARD_OFFSET_Y + grid_y * (self.GEM_SIZE + self.GRID_LINE_WIDTH) + self.GEM_SIZE / 2
        color = self.GEM_COLORS[gem_type - 1]
        
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'life': self.np_random.integers(15, 26),
                'color': color,
                'size': self.np_random.integers(2, 6)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 25.0))))
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Puzzle Environment")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    print("--- GAME RESET ---")
                elif event.key == pygame.K_q: # Quit on 'q' key
                    terminated = True

        action = [movement, space, shift]
        
        # In manual play, we only step when an action is taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Moves: {info['valid_moves']}, Terminated: {terminated}")
        else: # If no action, just re-render the current state
             obs = env._get_observation()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for playability
        
    print("Game Over!")
    pygame.quit()