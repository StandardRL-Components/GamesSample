
# Generated: 2025-08-27T18:40:52.946445
# Source Brief: brief_01916.md
# Brief Index: 1916

        
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
        "Controls: Arrow keys to move cursor. Space to reveal a tile. Shift to flag/unflag a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Navigate a grid, revealing tiles while avoiding hidden mines. Use logic to clear the entire field."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 10
    NUM_MINES = 15
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (50, 60, 70)
    COLOR_TILE_HIDDEN = (120, 130, 140)
    COLOR_TILE_REVEALED = (60, 70, 80)
    COLOR_TILE_EXPLODED = (255, 80, 80)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_FLAG = (255, 100, 100)
    
    # Color map for numbers based on mine count
    COLOR_NUMBERS = {
        1: (100, 150, 255),
        2: (100, 200, 100),
        3: (255, 100, 100),
        4: (150, 100, 255),
        5: (255, 150, 50),
        6: (100, 200, 200),
        7: (220, 220, 220),
        8: (180, 180, 180),
    }

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
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game constants
        self.screen_width, self.screen_height = self.screen.get_size()
        self.grid_area_size = self.screen_height - 20
        self.tile_size = self.grid_area_size // self.GRID_SIZE
        self.grid_offset_x = (self.screen_width - self.grid_area_size) // 2
        self.grid_offset_y = (self.screen_height - self.grid_area_size) // 2

        # Initialize state variables
        self.mine_grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.adjacency_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.mine_hit_pos = None
        self.particles = []

        self.reset()
        
        # Self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.mine_hit_pos = None
        self.particles = []

        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        
        # Initialize grids
        self.revealed_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), False, dtype=bool)
        self.flagged_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), False, dtype=bool)
        
        # Place mines
        self.mine_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), False, dtype=bool)
        flat_indices = self.np_random.choice(self.GRID_SIZE**2, self.NUM_MINES, replace=False)
        row_indices, col_indices = np.unravel_index(flat_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.mine_grid[row_indices, col_indices] = True
        
        # Pre-calculate adjacency counts
        self.adjacency_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if not self.mine_grid[r, c]:
                    self.adjacency_grid[r, c] = self._count_adjacent_mines(r, c)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1

        reward = 0
        
        # 1. Handle Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_SIZE - 1)

        # 2. Handle Flagging (Shift)
        if shift_pressed and not self.revealed_grid[self.cursor_pos[1], self.cursor_pos[0]]:
            self.flagged_grid[self.cursor_pos[1], self.cursor_pos[0]] = not self.flagged_grid[self.cursor_pos[1], self.cursor_pos[0]]

        # 3. Handle Revealing (Space)
        if space_pressed:
            r, c = self.cursor_pos[1], self.cursor_pos[0]
            if not self.revealed_grid[r, c] and not self.flagged_grid[r, c]:
                if self.mine_grid[r, c]:
                    # Game Over: Hit a mine
                    self.game_over = True
                    self.win = False
                    self.mine_hit_pos = (c, r)
                    self._create_explosion(c, r)
                    reward = -100.0
                else:
                    # Revealed a safe tile
                    reward += self._flood_fill_reveal(r, c)
        
        self.score += reward
        self.steps += 1
        
        # 4. Check for Win Condition
        if not self.game_over:
            revealed_count = np.sum(self.revealed_grid)
            if revealed_count == self.GRID_SIZE**2 - self.NUM_MINES:
                self.game_over = True
                self.win = True
                self.score += 100.0 # Add win bonus to the final score
                reward += 100.0

        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _count_adjacent_mines(self, r, c):
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and self.mine_grid[nr, nc]:
                    count += 1
        return count

    def _flood_fill_reveal(self, r_start, c_start):
        # Use a queue for non-recursive flood fill
        queue = [(r_start, c_start)]
        visited = set(queue)
        reward = 0
        
        while queue:
            r, c = queue.pop(0)
            
            if self.revealed_grid[r, c] or self.flagged_grid[r, c]:
                continue
            
            self.revealed_grid[r, c] = True
            
            adjacency_count = self.adjacency_grid[r, c]
            if adjacency_count == 0:
                reward += -0.2 # Discourage revealing empty space
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and (nr, nc) not in visited:
                            queue.append((nr, nc))
                            visited.add((nr, nc))
            else:
                reward += 1.0 # Standard reward for a safe tile
                
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update and draw particles
        self._update_particles()

        # Draw grid and tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.tile_size,
                    self.grid_offset_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                )
                
                is_revealed = self.revealed_grid[r, c]
                is_mine = self.mine_grid[r, c]
                is_flagged = self.flagged_grid[r, c]

                if self.game_over and is_mine and not is_flagged:
                    is_revealed = True # Reveal all mines on game over

                if is_revealed:
                    if is_mine:
                        color = self.COLOR_TILE_EXPLODED if (c, r) == self.mine_hit_pos else (150, 60, 60)
                        pygame.draw.rect(self.screen, color, rect)
                        # Draw mine symbol
                        pygame.draw.circle(self.screen, self.COLOR_BG, rect.center, self.tile_size // 4)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                        num = self.adjacency_grid[r, c]
                        if num > 0:
                            self._draw_text(str(num), rect.center, self.font_medium, self.COLOR_NUMBERS.get(num, self.COLOR_TEXT))
                else:
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect)
                    if is_flagged:
                        # Draw flag
                        p1 = (rect.centerx, rect.top + 5)
                        p2 = (rect.right - 5, rect.centery - 5)
                        p3 = (rect.centerx, rect.centery + 5)
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [p1, p2, p3])
                        pygame.draw.line(self.screen, self.COLOR_FLAG, p1, (rect.centerx, rect.bottom - 5), 2)


        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.tile_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.tile_size, self.grid_offset_y + self.grid_area_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.tile_size)
            end_pos = (self.grid_offset_x + self.grid_area_size, self.grid_offset_y + i * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.tile_size,
            self.grid_offset_y + self.cursor_pos[1] * self.tile_size,
            self.tile_size, self.tile_size
        )
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 # 0 to 1
        width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width, border_radius=2)
        
    def _render_ui(self):
        # Draw score
        score_text = f"Score: {self.score:.1f}"
        self._draw_text(score_text, (10, 10), self.font_small, self.COLOR_TEXT, align="topleft")
        
        # Draw steps
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(steps_text, (self.screen_width - 10, 10), self.font_small, self.COLOR_TEXT, align="topright")
        
        # Draw remaining mines
        flags_placed = np.sum(self.flagged_grid)
        mines_left_text = f"Mines: {self.NUM_MINES - flags_placed}"
        self._draw_text(mines_left_text, (10, 35), self.font_small, self.COLOR_TEXT, align="topleft")

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else self.COLOR_TILE_EXPLODED
            self._draw_text(message, self.screen.get_rect().center, self.font_large, color)
            
    def _create_explosion(self, c, r):
        # Sound placeholder: # pygame.mixer.Sound("explosion.wav").play()
        center_x = self.grid_offset_x + c * self.tile_size + self.tile_size // 2
        center_y = self.grid_offset_y + r * self.tile_size + self.tile_size // 2
        
        # Main expanding blast
        self.particles.append({
            "pos": [center_x, center_y], "vel": [0,0], "radius": 5, "max_radius": self.tile_size * 0.8,
            "color": (255, 255, 100), "type": "blast", "life": 15
        })
        
        # Debris particles
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": random.randint(2, 4),
                "color": random.choice([(100,100,100), (80,80,80), (120,120,120)]),
                "type": "debris", "life": random.randint(20, 40)
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            
            if p["type"] == "blast":
                p["radius"] += (p["max_radius"] - p["radius"]) * 0.2
                alpha = int(255 * (p["life"] / 15))
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), (*p["color"], alpha))
            else: # debris
                pygame.draw.circle(self.screen, p["color"], p["pos"], p["radius"])
                
    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "win": self.win,
            "mines_remaining": self.NUM_MINES - np.sum(self.flagged_grid)
        }
        
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode(obs.shape[1::-1]) # (640, 400)
    pygame.display.set_caption("Minesweeper Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping from keyboard ---
        movement = 0 # no-op
        space_pressed = 0
        shift_pressed = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_pressed = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed = 1
                if event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Only step if an action is taken (since auto_advance is False)
        action = [movement, space_pressed, shift_pressed]
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("--- GAME OVER --- Press 'R' to reset.")
        
        # --- Rendering ---
        # The environment's observation is already a rendered frame
        # We just need to blit it to the display screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()