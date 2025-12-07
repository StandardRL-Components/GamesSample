# Generated: 2025-08-27T16:53:18.372771
# Source Brief: brief_01360.md
# Brief Index: 1360

        
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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a tile. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic mine-sweeping puzzle. Reveal all safe tiles while avoiding the hidden mines. Numbers on revealed tiles indicate how many mines are adjacent."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = (8, 8)
    NUM_MINES = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (50, 60, 70)
    COLOR_TILE_HIDDEN = (70, 80, 95)
    COLOR_TILE_REVEALED = (110, 125, 145)
    COLOR_TILE_FLAG = (255, 200, 0)
    COLOR_MINE = (255, 80, 80)
    COLOR_CURSOR = (100, 200, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_NUMBERS = [
        (0, 0, 0, 0), # 0 is transparent
        (80, 150, 255),  # 1
        (80, 200, 80),   # 2
        (255, 80, 80),   # 3
        (150, 80, 220),  # 4
        (220, 130, 0),   # 5
        (0, 200, 200),   # 6
        (200, 0, 200),   # 7
        (100, 100, 100)  # 8
    ]


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_tile = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Verdana", 48, bold=True)

        # Game grid layout
        self.grid_rows, self.grid_cols = self.GRID_SIZE
        self.tile_size = 36
        self.grid_width = self.grid_cols * self.tile_size
        self.grid_height = self.grid_rows * self.tile_size
        self.grid_origin_x = (self.width - self.grid_width) // 2
        self.grid_origin_y = (self.height - self.grid_height) // 2
        
        # Initialize state variables
        self.cursor_pos = None
        self.mine_grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.number_grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.last_shift_held = False
        self.flags_placed = 0
        self.safe_tiles_revealed = 0
        self.total_safe_tiles = self.grid_rows * self.grid_cols - self.NUM_MINES
        self.particles = []

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.last_shift_held = False
        self.flags_placed = 0
        self.safe_tiles_revealed = 0
        self.particles = []

        self.cursor_pos = [self.grid_rows // 2, self.grid_cols // 2]

        # Create grids
        self.mine_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.revealed_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.flagged_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.number_grid = np.zeros(self.GRID_SIZE, dtype=int)

        self._place_mines()
        self._calculate_numbers()

        return self._get_observation(), self._get_info()

    def _place_mines(self):
        flat_indices = self.np_random.choice(
            self.grid_rows * self.grid_cols, self.NUM_MINES, replace=False
        )
        row_indices, col_indices = np.unravel_index(flat_indices, self.GRID_SIZE)
        self.mine_grid[row_indices, col_indices] = True
    
    def _calculate_numbers(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.mine_grid[r, c]:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and self.mine_grid[nr, nc]:
                            count += 1
                self.number_grid[r, c] = count

    def step(self, action):
        reward = 0
        
        if self.game_over:
            # If game is over, no actions have an effect, just return current state
            terminated = self._check_termination()
            return self._get_observation(), 0, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Detect rising edge for one-shot actions
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        r, c = self.cursor_pos

        # Action Priority: Reveal > Flag > Move
        if space_pressed:
            reward = self._handle_reveal(r, c)
        elif shift_pressed:
            reward = self._handle_flag(r, c)
        elif movement != 0:
            self._handle_move(movement)
            # No reward for just moving

        self.score += reward
        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_reveal(self, r, c):
        if self.revealed_grid[r, c] or self.flagged_grid[r, c]:
            return 0  # No action, no reward

        if self.mine_grid[r, c]:
            # Game over: hit a mine
            self.game_over = True
            self.win = False
            self.revealed_grid[r, c] = True # Reveal the mine
            self._create_explosion(r, c)
            # sfx: explosion
            return -100.0

        # Revealed a safe tile
        tiles_revealed_before = self.safe_tiles_revealed
        self._flood_fill(r, c)
        tiles_revealed_after = self.safe_tiles_revealed
        
        newly_revealed_count = tiles_revealed_after - tiles_revealed_before
        
        # Check for win condition
        if self.safe_tiles_revealed == self.total_safe_tiles:
            self.game_over = True
            self.win = True
            # sfx: win_jingle
            return 100.0 + newly_revealed_count # Win bonus + reward for revealing tiles
        
        # sfx: reveal_tile
        return float(newly_revealed_count) # +1 for each newly revealed tile

    def _flood_fill(self, r, c):
        q = deque([(r, c)])
        visited = set([(r, c)])
        
        while q:
            row, col = q.popleft()
            
            if self.revealed_grid[row, col] or self.flagged_grid[row, col]:
                continue
                
            self.revealed_grid[row, col] = True
            self.safe_tiles_revealed += 1
            
            if self.number_grid[row, col] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and (nr, nc) not in visited:
                            q.append((nr, nc))
                            visited.add((nr, nc))

    def _handle_flag(self, r, c):
        if self.revealed_grid[r, c]:
            return 0 # Cannot flag a revealed tile

        if self.flagged_grid[r, c]:
            self.flagged_grid[r, c] = False
            self.flags_placed -= 1
            # sfx: unflag
            return 0.1 # Small reward for removing a flag (recovering a resource)
        else:
            self.flagged_grid[r, c] = True
            self.flags_placed += 1
            # sfx: flag
            return -0.1 # Small penalty for placing a flag

    def _handle_move(self, movement):
        # sfx: cursor_move
        r, c = self.cursor_pos
        if movement == 1:  # Up
            self.cursor_pos[0] = (r - 1) % self.grid_rows
        elif movement == 2:  # Down
            self.cursor_pos[0] = (r + 1) % self.grid_rows
        elif movement == 3:  # Left
            self.cursor_pos[1] = (c - 1) % self.grid_cols
        elif movement == 4:  # Right
            self.cursor_pos[1] = (c + 1) % self.grid_cols

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_render_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                rect = pygame.Rect(
                    self.grid_origin_x + c * self.tile_size,
                    self.grid_origin_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                )
                
                # Draw main tile body
                if self.revealed_grid[r, c]:
                    if self.mine_grid[r, c]:
                        pygame.draw.rect(self.screen, self.COLOR_MINE, rect)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # Draw content on tile
                if self.revealed_grid[r, c]:
                    if self.mine_grid[r, c]:
                        # Draw mine symbol (circle)
                        center = rect.center
                        pygame.draw.circle(self.screen, (30,30,30), center, self.tile_size // 3)
                    elif self.number_grid[r, c] > 0:
                        num_text = str(self.number_grid[r, c])
                        color = self.COLOR_NUMBERS[self.number_grid[r, c]]
                        text_surf = self.font_tile.render(num_text, True, color)
                        text_rect = text_surf.get_rect(center=rect.center)
                        self.screen.blit(text_surf, text_rect)
                elif self.flagged_grid[r, c]:
                    # Draw flag symbol (triangle)
                    p1 = (rect.centerx, rect.top + 6)
                    p2 = (rect.centerx, rect.centery + 4)
                    p3 = (rect.left + 6, rect.centery - 2)
                    pygame.draw.polygon(self.screen, self.COLOR_TILE_FLAG, [p1, p2, p3])
                    pygame.draw.line(self.screen, self.COLOR_TILE_FLAG, (rect.centerx, rect.top + 6), (rect.centerx, rect.bottom - 6), 2)
        
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_origin_x + cursor_c * self.tile_size,
            self.grid_origin_y + cursor_r * self.tile_size,
            self.tile_size, self.tile_size
        )
        # Pulsing alpha effect for cursor
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_surf = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, (*self.COLOR_CURSOR, alpha), (0, 0, self.tile_size, self.tile_size), 3)
        self.screen.blit(cursor_surf, cursor_rect.topleft)

        # On game over, reveal all mines
        if self.game_over and not self.win:
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if self.mine_grid[r, c] and not self.revealed_grid[r,c]:
                        rect = pygame.Rect(
                            self.grid_origin_x + c * self.tile_size,
                            self.grid_origin_y + r * self.tile_size,
                            self.tile_size, self.tile_size
                        )
                        pygame.draw.circle(self.screen, (100,100,100), rect.center, self.tile_size // 4)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Flags remaining
        flags_text = f"FLAGS: {self.NUM_MINES - self.flags_placed}"
        flags_surf = self.font_main.render(flags_text, True, self.COLOR_TEXT)
        flags_rect = flags_surf.get_rect(topright=(self.width - 20, 20))
        self.screen.blit(flags_surf, flags_rect)
        
        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else self.COLOR_MINE
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.width / 2, self.height / 2))
            
            # Draw a semi-transparent background for the message
            bg_rect = msg_rect.inflate(40, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            
            # Create a valid RGBA color tuple. The original code used a generator expression
            # that could not be concatenated with a tuple, causing the error.
            bg_color = tuple(int(c * 0.5) for c in self.COLOR_BG) + (200,)
            bg_surf.fill(bg_color)
            
            self.screen.blit(bg_surf, bg_rect)
            
            self.screen.blit(msg_surf, msg_rect)

    def _create_explosion(self, r, c):
        center_x = self.grid_origin_x + c * self.tile_size + self.tile_size / 2
        center_y = self.grid_origin_y + r * self.tile_size + self.tile_size / 2
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(20, 40)
            color = random.choice([self.COLOR_MINE, (255,150,0), (200,200,200)])
            self.particles.append([center_x, center_y, vx, vy, lifetime, color])

    def _update_and_render_particles(self):
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime -= 1
            size = max(0, int(p[4] / 6))
            if size > 0:
                pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), size)
        self.particles = [p for p in self.particles if p[4] > 0]
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "flags_placed": self.flags_placed,
            "safe_tiles_revealed": self.safe_tiles_revealed,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `render_mode` to "human" for a visible window
    class HumanGameEnv(GameEnv):
        metadata = {"render_modes": ["rgb_array", "human"]}
        def __init__(self, render_mode="human"):
            super().__init__(render_mode)
            self.render_mode = render_mode
            if self.render_mode == "human":
                # Re-initialize screen for display
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Minesweeper Gym Environment")
        
        def _get_observation(self):
            # First, get the observation from the parent class (which renders to a surface)
            obs = super()._get_observation()
            if self.render_mode == "human":
                # The parent method drew on a surface, we just need to show it.
                # Pygame needs (width, height, channels) so we transpose back.
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
            return obs

    env = HumanGameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    print("--- RESET ---")

        if not terminated:
            keys = pygame.key.get_pressed()
            # Movement
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            # Actions
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            if np.any(action != 0):
                 # Only print when an action is taken to avoid spam
                 # print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                 pass

        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()