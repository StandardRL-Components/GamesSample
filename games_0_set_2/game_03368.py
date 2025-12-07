import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to clear a group of 3 or more matching tiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by selecting groups of 3 or more adjacent, matching-colored tiles. Run out of moves and you lose!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 6
    TILE_WIDTH = 80
    TILE_HEIGHT = 60
    UI_HEIGHT = 40
    SCREEN_WIDTH = GRID_WIDTH * TILE_WIDTH # 640
    SCREEN_HEIGHT = GRID_HEIGHT * TILE_HEIGHT + UI_HEIGHT # 360 + 40 = 400
    
    MAX_MOVES = 20
    MIN_MATCH_SIZE = 3
    
    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (50, 50, 60)
    
    COLORS = [
        (255, 87, 87),    # Red
        (87, 255, 87),    # Green
        (87, 137, 255),   # Blue
        (255, 255, 87),   # Yellow
        (200, 87, 255),   # Purple
    ]
    
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid = None
        self.cursor_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.animation_tick = 0
        
        # This call to reset() is necessary to initialize the state
        # but since it can be slow, it's good practice to allow it to be seeded
        # The validation below will use a default state.
        # self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use Python's random for game logic, numpy's for anything else if needed
        # self.np_random is available from super().reset() if needed
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        self.animation_tick = 0

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0.0
        
        self.animation_tick += 1
        self._update_particles()

        # --- Handle Cursor Movement ---
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH
            
        # --- Handle Action ---
        if space_held == 1 and not self.game_over:
            self.steps += 1
            cursor_x, cursor_y = self.cursor_pos
            
            connected_tiles = self._find_connected_tiles(cursor_x, cursor_y)
            
            if len(connected_tiles) >= self.MIN_MATCH_SIZE:
                # Successful match
                num_cleared = len(connected_tiles)
                reward = min(10.0, float(num_cleared)) # Cap non-terminal reward
                self.score += num_cleared
                
                # Clear tiles and create particles
                tile_color = self.COLORS[self.grid[cursor_y][cursor_x]]
                for x, y in connected_tiles:
                    self.grid[y][x] = -1 # Mark as empty
                    self._create_particles(x, y, tile_color)
                
                # Apply gravity and refill
                self._handle_gravity_and_refill()
                
                # Check for win condition (board cleared)
                if all(all(tile == -1 for tile in row) for row in self.grid):
                    self.game_over = True
                    self.win = True
                    reward += 100.0
            else:
                # Failed match
                self.moves_left -= 1
                reward = -1.0
        
        # --- Check Termination ---
        if not self.game_over and self.moves_left <= 0:
            # Check if there are any valid moves left. If not, game over.
            has_possible_move = False
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if self.grid[y][x] != -1:
                        if len(self._find_connected_tiles(x,y)) >= self.MIN_MATCH_SIZE:
                            has_possible_move = True
                            break
                if has_possible_move:
                    break
            
            if not has_possible_move:
                self.game_over = True
                self.win = False
                reward = -100.0
            elif self.moves_left <= 0:
                self.game_over = True
                self.win = False
                reward = -100.0
                
        terminated = self.game_over
        truncated = False # No truncation condition in this game
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, channels), we need (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
        }

    # --- Game Logic Helpers ---
    def _generate_board(self):
        while True:
            self.grid = [[random.randint(0, len(self.COLORS) - 1) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
            
            has_possible_move = False
            visited_for_check = set()
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if (x, y) not in visited_for_check:
                        group = self._find_connected_tiles(x, y)
                        if len(group) >= self.MIN_MATCH_SIZE:
                            has_possible_move = True
                            break
                        visited_for_check.update(group)
                if has_possible_move:
                    break
            
            if has_possible_move:
                break

    def _find_connected_tiles(self, start_x, start_y):
        if not (0 <= start_x < self.GRID_WIDTH and 0 <= start_y < self.GRID_HEIGHT):
            return []
            
        target_color = self.grid[start_y][start_x]
        if target_color == -1:
            return []

        q = [(start_x, start_y)]
        visited = set(q)
        group = []

        while q:
            x, y = q.pop(0)
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny][nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _handle_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] == -1:
                    empty_slots.append(y)
                elif empty_slots:
                    new_y = empty_slots.pop(0)
                    self.grid[new_y][x] = self.grid[y][x]
                    self.grid[y][x] = -1
                    empty_slots.append(y)
            
            for y in empty_slots:
                self.grid[y][x] = random.randint(0, len(self.COLORS) - 1)

    def _create_particles(self, grid_x, grid_y, color):
        px = grid_x * self.TILE_WIDTH + self.TILE_WIDTH / 2
        py = grid_y * self.TILE_HEIGHT + self.TILE_HEIGHT / 2 + self.UI_HEIGHT
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            if p['life'] > 0:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1 # Gravity
                p['life'] -= 1
                active_particles.append(p)
        self.particles = active_particles

    # --- Rendering Methods ---
    def render(self):
        return self._get_observation()

    def _render_game(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    x * self.TILE_WIDTH, 
                    y * self.TILE_HEIGHT + self.UI_HEIGHT, 
                    self.TILE_WIDTH, 
                    self.TILE_HEIGHT
                )
                
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                color_idx = self.grid[y][x]
                if color_idx != -1:
                    color = self.COLORS[color_idx]
                    inner_rect = rect.inflate(-8, -8)
                    
                    pygame.gfxdraw.box(self.screen, inner_rect, color)
                    border_color = tuple(max(0, c - 40) for c in color)
                    pygame.gfxdraw.rectangle(self.screen, inner_rect, border_color)
        
        self._draw_cursor()
        self._update_and_draw_particles()

    def _draw_cursor(self):
        if self.game_over: return
        
        x, y = self.cursor_pos
        rect = pygame.Rect(
            x * self.TILE_WIDTH, 
            y * self.TILE_HEIGHT + self.UI_HEIGHT, 
            self.TILE_WIDTH, 
            self.TILE_HEIGHT
        )
        
        pulse = (math.sin(self.animation_tick * 0.2) + 1) / 2
        thickness = int(2 + pulse * 3)
        
        pygame.draw.rect(self.screen, self.COLOR_WHITE, rect, thickness, border_radius=4)
        
    def _update_and_draw_particles(self):
        for p in self.particles:
            size = int(max(0, p['life'] / 8))
            if size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.rect(self.screen, p['color'], (pos[0] - size//2, pos[1] - size//2, size, size))

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_WHITE)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            self._draw_game_over()

    def _draw_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "You Win!" if self.win else "Game Over"
        text_surf = self.font_large.render(message, True, self.COLOR_WHITE)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)
        
    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    # The main loop is for interactive play and debugging, not for the headless environment
    # It requires a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Tile Matcher")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        movement = 0
        space_held = 0
        
        # This is a key-down event driven game, not key-held
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
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
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    # After reset, render the new state
                    obs = env.render()
                    continue
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                
                action = [movement, space_held, 0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                if reward != 0:
                    print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")
                if terminated:
                    print("--- GAME OVER ---")
                    print(f"Final Score: {info['score']}")
                    # Render the final game over screen
                    obs = env.render()

        # Render the current state from the environment's observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)

    env.close()