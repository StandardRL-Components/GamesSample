
# Generated: 2025-08-28T05:26:57.727901
# Source Brief: brief_02629.md
# Brief Index: 2629

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move cursor. Press space to reveal a square. Press shift to place/remove a flag."
    )

    # User-facing description of the game
    game_description = (
        "A classic puzzle game of logic and deduction. Clear the board by revealing all safe squares while avoiding the hidden mines."
    )

    # The game is turn-based, so it should only advance when an action is received.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 9
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (35, 35, 45)
        self.COLOR_GRID_LINES = (60, 60, 70)
        self.COLOR_UNREVEALED = (130, 140, 150)
        self.COLOR_REVEALED = (190, 200, 210)
        self.COLOR_CURSOR = (255, 220, 0)
        self.COLOR_FLAG = (60, 220, 120)
        self.COLOR_MINE_EXPLODED = (255, 70, 70)
        self.COLOR_MINE_ICON = (20, 20, 20)
        self.COLOR_TEXT_UI = (230, 230, 230)
        self.COLOR_WIN = (60, 220, 120)
        self.COLOR_LOSE = (255, 70, 70)
        self.NUMBER_COLORS = [
            self.COLOR_REVEALED,      # 0 is not drawn
            (0, 128, 255),            # 1: Blue
            (0, 192, 0),              # 2: Green
            (255, 0, 0),              # 3: Red
            (0, 0, 128),              # 4: Dark Blue
            (128, 0, 0),              # 5: Brown
            (0, 192, 192),            # 6: Cyan
            (0, 0, 0),                # 7: Black
            (128, 128, 128),          # 8: Grey
        ]

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_ui = pygame.font.SysFont("Consolas", 18)
            self.font_status = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 30, bold=True)
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_status = pygame.font.SysFont(None, 60, bold=True)

        # Rendering properties
        self.cell_size = 36
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_height) // 2 + 10

        # Initialize state variables
        self.mine_grid = None
        self.revealed_grid = None
        self.flag_grid = None
        self.number_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.safe_squares_total = 0
        self.safe_squares_revealed = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize grids
        self.mine_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.revealed_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.flag_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.number_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Place mines
        self._place_mines()
        
        # Calculate numbers
        self._calculate_numbers()
        
        self.safe_squares_total = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
        self.safe_squares_revealed = 0
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _place_mines(self):
        mine_count = 0
        while mine_count < self.NUM_MINES:
            x = self.np_random.integers(0, self.GRID_SIZE)
            y = self.np_random.integers(0, self.GRID_SIZE)
            if not self.mine_grid[y, x]:
                self.mine_grid[y, x] = True
                mine_count += 1

    def _calculate_numbers(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.mine_grid[y, x]:
                    continue
                count = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.mine_grid[ny, nx]:
                            count += 1
                self.number_grid[y, x] = count

    def step(self, action):
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1 and not self.prev_space_held
        shift_pressed = action[2] == 1 and not self.prev_shift_held
        
        self.prev_space_held = action[1] == 1
        self.prev_shift_held = action[2] == 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # 1. Update cursor position
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
            
        cx, cy = self.cursor_pos
        
        # 2. Handle reveal action (Space)
        if space_pressed:
            if self.flag_grid[cy, cx] or self.revealed_grid[cy, cx]:
                reward -= 0.05 # Penalty for ineffective action
            elif self.mine_grid[cy, cx]:
                self.game_over = True
                self.win_state = False
                terminated = True
                reward = -100
                self.revealed_grid[cy, cx] = True
                self._create_explosion(cx, cy)
                # sound: explosion
            else:
                revealed_count = self._reveal_square(cx, cy)
                reward += revealed_count # +1 for each newly revealed safe square
        
        # 3. Handle flag action (Shift)
        if shift_pressed:
            if not self.revealed_grid[cy, cx]:
                is_mine = self.mine_grid[cy, cx]
                if self.flag_grid[cy, cx]:
                    self.flag_grid[cy, cx] = False
                    if not is_mine: reward += 0.1 # Correctly removing flag from safe spot
                    # sound: unflag
                else:
                    self.flag_grid[cy, cx] = True
                    if not is_mine: reward -= 0.1 # Incorrectly flagging a safe spot
                    # sound: flag
            else:
                reward -= 0.05 # Penalty for ineffective action

        # 4. Check for win condition
        if not self.game_over and self.safe_squares_revealed == self.safe_squares_total:
            self.game_over = True
            self.win_state = True
            terminated = True
            reward = 100
            # sound: win
            
        # 5. Check for max steps termination
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _reveal_square(self, x, y):
        # Flood-fill reveal for empty squares
        q = deque([(x, y)])
        visited = set([(x, y)])
        newly_revealed = 0
        
        while q:
            cx, cy = q.popleft()
            
            if self.revealed_grid[cy, cx] or self.flag_grid[cy, cx]:
                continue
            
            self.revealed_grid[cy, cx] = True
            self.safe_squares_revealed += 1
            newly_revealed += 1
            # sound: reveal
            
            if self.number_grid[cy, cx] == 0:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                            q.append((nx, ny))
                            visited.add((nx, ny))
                            
        return newly_revealed

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                is_revealed = self.revealed_grid[y, x]
                is_flagged = self.flag_grid[y, x]
                is_mine = self.mine_grid[y, x]
                number = self.number_grid[y, x]

                # Draw base square
                if is_revealed:
                    bg_color = self.COLOR_MINE_EXPLODED if is_mine else self.COLOR_REVEALED
                    pygame.draw.rect(self.screen, bg_color, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                
                # Draw content
                if self.game_over and is_mine: # Show all mines on game over
                    if not is_revealed:
                         pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect) # Keep un-hit mines on gray
                    self._draw_mine(rect)
                elif is_revealed and not is_mine and number > 0:
                    self._draw_number(number, rect)
                elif not is_revealed and is_flagged:
                    self._draw_flag(rect)
                    
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _draw_number(self, number, rect):
        num_text = self.font_main.render(str(number), True, self.NUMBER_COLORS[number])
        text_rect = num_text.get_rect(center=rect.center)
        self.screen.blit(num_text, text_rect)

    def _draw_flag(self, rect):
        # Simple flag: pole and triangle
        center_x, center_y = rect.center
        pygame.draw.line(self.screen, self.COLOR_FLAG, (center_x, rect.top + 5), (center_x, rect.bottom - 5), 2)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [(center_x, rect.top + 5), (center_x + 10, rect.top + 10), (center_x, rect.top + 15)])
        
    def _draw_mine(self, rect):
        pygame.draw.circle(self.screen, self.COLOR_MINE_ICON, rect.center, self.cell_size // 3)
        for i in range(4):
            angle = i * math.pi / 2 + math.pi / 4
            start_pos = (rect.centerx + math.cos(angle) * self.cell_size * 0.2, 
                         rect.centery + math.sin(angle) * self.cell_size * 0.2)
            end_pos = (rect.centerx + math.cos(angle) * self.cell_size * 0.45,
                       rect.centery + math.sin(angle) * self.cell_size * 0.45)
            pygame.draw.line(self.screen, self.COLOR_MINE_ICON, start_pos, end_pos, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Mines remaining
        flags_placed = np.sum(self.flag_grid)
        mines_left = self.NUM_MINES - flags_placed
        mines_text = self.font_ui.render(f"Mines: {mines_left}", True, self.COLOR_TEXT_UI)
        mines_rect = mines_text.get_rect(right=self.WIDTH - 10, top=10)
        self.screen.blit(mines_text, mines_rect)
        
        # Game over status
        if self.game_over:
            if self.win_state:
                status_text = self.font_status.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                status_text = self.font_status.render("GAME OVER", True, self.COLOR_LOSE)
            
            status_rect = status_text.get_rect(center=(self.WIDTH // 2, self.grid_offset_y // 2))
            # Background for text
            bg_rect = status_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, bg_rect, 2, border_radius=5)
            self.screen.blit(status_text, status_rect)

    def _create_explosion(self, grid_x, grid_y):
        center_x = self.grid_offset_x + grid_x * self.cell_size + self.cell_size // 2
        center_y = self.grid_offset_y + grid_y * self.cell_size + self.cell_size // 2
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            particle = {
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.uniform(20, 40),
                "radius": self.np_random.uniform(2, 5),
                "color": random.choice([(255, 70, 70), (255, 150, 0), (200, 200, 200)])
            }
            self.particles.append(particle)

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] *= 0.95
            if p["life"] > 0 and p["radius"] > 0.5:
                alpha = int(255 * (p["life"] / 40))
                color = p["color"]
                
                surf = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*color, alpha), (p["radius"], p["radius"]), p["radius"])
                self.screen.blit(surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "safe_squares_revealed": self.safe_squares_revealed,
            "flags_placed": np.sum(self.flag_grid),
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Environment")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Print controls
    print("\n" + "="*30)
    print(" Minesweeper Gym Environment")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_ESCAPE:
                    terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        # In a turn-based game, we only step if an action is taken.
        # For manual play, this means stepping on every keydown.
        # For an agent, it would step on every call.
        # The 'movement' action is momentary, so we step even if it's the only thing.
        if any(action):
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {term}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit frame rate for manual play
        
    print("Game Over!")
    pygame.time.wait(2000)
    env.close()