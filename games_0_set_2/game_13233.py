import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:57:18.809060
# Source Brief: brief_03233.md
# Brief Index: 3233
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Drop numbered pieces into the grid to form simple equations. "
        "Solve equations that match the target number to score points and clear blocks."
    )
    user_guide = "Controls: ←→ to move the falling piece."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (60, 80, 110)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TARGET = (255, 190, 0)
    COLOR_FALLING_PIECE = (50, 150, 255)
    COLOR_FALLING_PIECE_GLOW = (50, 150, 255, 50)
    COLOR_SOLVE_GOOD = (0, 255, 120)
    COLOR_SOLVE_BAD = (255, 80, 80)
    COLOR_MOMENTUM_FLAME = [(255, 230, 0), (255, 150, 0), (255, 60, 0)]

    # Game settings
    GRID_SIZE = 3
    GRID_CELL_SIZE = 60
    GRID_LINE_WIDTH = 4
    MAX_STEPS = 1000
    WIN_CONDITION = 10
    FALL_SPEED = 4.0
    MOMENTUM_THRESHOLD = 3
    
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
        
        try:
            self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
            self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 30)
            self.font_large = pygame.font.Font(None, 54)

        self.grid_width = self.GRID_SIZE * self.GRID_CELL_SIZE
        self.grid_height = self.GRID_SIZE * self.GRID_CELL_SIZE
        self.grid_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 - 20

        # These attributes are defined in reset()
        self.grid = None
        self.falling_piece = None
        self.target_number = None
        self.score = None
        self.steps = None
        self.equations_solved = None
        self.consecutive_solves = None
        self.momentum_active = None
        self.game_over = None
        self.particles = None
        self.highlight_info = None
        self.animated_score = None
        
        # self.reset() # reset() is called by the wrapper
        # self.validate_implementation() # this is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=int)
        self.score = 0
        self.animated_score = 0
        self.steps = 0
        self.equations_solved = 0
        self.consecutive_solves = 0
        self.momentum_active = False
        self.game_over = False
        self.particles = []
        self.highlight_info = None # {'coords': [], 'color': (), 'timer': 0}
        
        self._generate_target_number()
        self._spawn_falling_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small penalty for each step to encourage speed
        
        self._handle_action(action)
        reward += self._update_game_state()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if self.equations_solved >= self.WIN_CONDITION and not self.game_over:
            reward += 50
            self.game_over = True
            terminated = True
            # sfx: win_game
        elif self._is_grid_full() and not self.game_over:
            reward -= 50
            self.game_over = True
            terminated = True
            # sfx: lose_game
        elif self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        
        if movement == 3: # Left
            self.falling_piece['col'] = max(0, self.falling_piece['col'] - 1)
        elif movement == 4: # Right
            self.falling_piece['col'] = min(self.GRID_SIZE - 1, self.falling_piece['col'] + 1)
        
        target_x = self.grid_x + self.falling_piece['col'] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
        self.falling_piece['target_x'] = target_x

    def _update_game_state(self):
        reward = 0
        self.falling_piece['y'] += self.FALL_SPEED
        
        col = self.falling_piece['col']
        highest_empty_row = -1
        for r in range(self.GRID_SIZE - 1, -1, -1):
            if self.grid[r, col] == -1:
                highest_empty_row = r
                break

        if highest_empty_row != -1:
            landing_y = self.grid_y + highest_empty_row * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
            if self.falling_piece['y'] >= landing_y:
                self.grid[highest_empty_row, col] = self.falling_piece['value']
                self.falling_piece['y'] = landing_y # Snap to position
                self._create_particles((self.falling_piece['x'], self.falling_piece['y']), self.COLOR_FALLING_PIECE, 10, 'land')
                # sfx: place_piece
                
                reward += self._process_chain_reactions()

                if not self.game_over:
                    self._spawn_falling_piece()
        else: # Column is full
            landing_y = self.grid_y - self.GRID_CELL_SIZE // 2
            if self.falling_piece['y'] >= landing_y:
                # Piece lands on top of a full column, game over
                self.game_over = True
                reward -= 50

        return reward

    def _process_chain_reactions(self):
        total_reward = 0
        
        while True:
            made_change = False
            coords_to_clear = set()
            new_highlights = []
            
            # Check all rows and columns for equations
            for i in range(self.GRID_SIZE):
                # Row check
                row_coords = [(i, c) for c in range(self.GRID_SIZE)]
                is_eq, result, op_str, eq_coords = self._check_line(row_coords)
                if is_eq:
                    if result == self.target_number:
                        coords_to_clear.update(eq_coords)
                        new_highlights.append({'coords': eq_coords, 'color': self.COLOR_SOLVE_GOOD, 'timer': 60})
                    else:
                        total_reward += 1.0
                        self.score += 1
                        self.consecutive_solves = 0
                        self.momentum_active = False
                        new_highlights.append({'coords': eq_coords, 'color': self.COLOR_SOLVE_BAD, 'timer': 30})

                # Column check
                col_coords = [(r, i) for r in range(self.GRID_SIZE)]
                is_eq, result, op_str, eq_coords = self._check_line(col_coords)
                if is_eq:
                    if result == self.target_number:
                        coords_to_clear.update(eq_coords)
                        new_highlights.append({'coords': eq_coords, 'color': self.COLOR_SOLVE_GOOD, 'timer': 60})
                    else:
                        total_reward += 1.0
                        self.score += 1
                        self.consecutive_solves = 0
                        self.momentum_active = False
                        new_highlights.append({'coords': eq_coords, 'color': self.COLOR_SOLVE_BAD, 'timer': 30})
            
            if coords_to_clear:
                made_change = True
                # sfx: solve_equation
                
                # Calculate score and reward for target solve
                points = 15 * (1.5 if self.momentum_active else 1.0)
                total_reward += 15
                self.score += points
                self.equations_solved += 1
                self.consecutive_solves += 1
                
                if self.consecutive_solves >= self.MOMENTUM_THRESHOLD and not self.momentum_active:
                    self.momentum_active = True
                    total_reward += 5 # Bonus for activating momentum
                    # sfx: momentum_start
                
                for r, c in coords_to_clear:
                    if self.grid[r, c] != -1:
                        pos = (self.grid_x + c * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2,
                               self.grid_y + r * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2)
                        self._create_particles(pos, self.COLOR_SOLVE_GOOD, 30, 'solve')
                        self.grid[r, c] = -1

                self._apply_gravity()
                self._generate_target_number()
                self.highlight_info = new_highlights[0] # Show first good highlight
            
            else: # No target-solving equations found
                if new_highlights:
                    self.highlight_info = new_highlights[0]
                
            if not made_change:
                break
        
        return total_reward

    def _check_line(self, line_coords):
        vals = [self.grid[r, c] for r, c in line_coords]
        if -1 in vals:
            return False, None, None, []
        
        a, b, c = vals
        # A op B = C
        if a + b == c: return True, c, f"{a}+{b}={c}", line_coords
        if a - b == c: return True, c, f"{a}-{b}={c}", line_coords
        if a * b == c: return True, c, f"{a}*{b}={c}", line_coords
        if b != 0 and a / b == c: return True, c, f"{a}/{b}={c}", line_coords
        
        return False, None, None, []

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _spawn_falling_piece(self):
        self.falling_piece = {
            'value': self.np_random.integers(0, 10),
            'col': self.GRID_SIZE // 2,
            'x': self.grid_x + (self.GRID_SIZE // 2) * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2,
            'target_x': self.grid_x + (self.GRID_SIZE // 2) * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2,
            'y': self.grid_y - self.GRID_CELL_SIZE / 2,
            'size': self.GRID_CELL_SIZE * 0.4,
            'angle': 0
        }

    def _generate_target_number(self):
        a, b = self.np_random.integers(1, 10, size=2)
        op = self.np_random.choice(['+', '-', '*'])
        if op == '+': self.target_number = a + b
        elif op == '-': self.target_number = abs(a - b)
        elif op == '*': self.target_number = a * b
        if self.target_number == 0: self.target_number = self.np_random.integers(1, 10)

    def _is_grid_full(self):
        return np.all(self.grid != -1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._update_and_render_particles()
        self._render_grid_and_numbers()
        if self.highlight_info and self.highlight_info['timer'] > 0:
            self._render_highlight()
            self.highlight_info['timer'] -= 1
        if not self.game_over:
            self._render_falling_piece()

    def _render_grid_and_numbers(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_x + i * self.GRID_CELL_SIZE, self.grid_y)
            end_pos = (self.grid_x + i * self.GRID_CELL_SIZE, self.grid_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)
            # Horizontal
            start_pos = (self.grid_x, self.grid_y + i * self.GRID_CELL_SIZE)
            end_pos = (self.grid_x + self.grid_width, self.grid_y + i * self.GRID_CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)
        
        # Draw numbers in grid
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] != -1:
                    num_str = str(self.grid[r, c])
                    text_surf = self.font_large.render(num_str, True, self.COLOR_TEXT)
                    text_rect = text_surf.get_rect(center=(
                        self.grid_x + c * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2,
                        self.grid_y + r * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
                    ))
                    self.screen.blit(text_surf, text_rect)

    def _render_falling_piece(self):
        fp = self.falling_piece
        # Interpolate x position for smooth movement
        fp['x'] += (fp['target_x'] - fp['x']) * 0.4
        
        # Glow effect
        glow_radius = int(fp['size'] * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_FALLING_PIECE_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (int(fp['x'] - glow_radius), int(fp['y'] - glow_radius)))
        
        # Main piece
        pygame.gfxdraw.filled_circle(self.screen, int(fp['x']), int(fp['y']), int(fp['size']), self.COLOR_FALLING_PIECE)
        pygame.gfxdraw.aacircle(self.screen, int(fp['x']), int(fp['y']), int(fp['size']), self.COLOR_FALLING_PIECE)
        
        # Number on piece
        num_str = str(fp['value'])
        text_surf = self.font_large.render(num_str, True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=(int(fp['x']), int(fp['y'])))
        self.screen.blit(text_surf, text_rect)

    def _render_highlight(self):
        info = self.highlight_info
        alpha = int(150 * (info['timer'] / 60.0))
        color = info['color'] + (alpha,)
        
        for r, c in info['coords']:
            rect = pygame.Rect(
                self.grid_x + c * self.GRID_CELL_SIZE,
                self.grid_y + r * self.GRID_CELL_SIZE,
                self.GRID_CELL_SIZE, self.GRID_CELL_SIZE
            )
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # Target Number
        target_text = self.font_medium.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (self.grid_x, 30))
        target_num_text = self.font_large.render(str(self.target_number), True, self.COLOR_TARGET)
        self.screen.blit(target_num_text, (self.grid_x + target_text.get_width() + 15, 15))

        # Score
        if abs(self.score - self.animated_score) > 0.1:
            self.animated_score += (self.score - self.animated_score) * 0.2
        else:
            self.animated_score = self.score
        
        score_text = self.font_medium.render("SCORE", True, self.COLOR_TEXT)
        score_pos_y = self.grid_y + self.grid_height + 25
        self.screen.blit(score_text, (self.grid_x, score_pos_y))
        score_num_text = self.font_large.render(str(int(self.animated_score)), True, self.COLOR_TEXT)
        self.screen.blit(score_num_text, (self.grid_x + score_text.get_width() + 15, score_pos_y - 10))

        # Solved
        solved_str = f"SOLVED: {self.equations_solved}/{self.WIN_CONDITION}"
        solved_text = self.font_small.render(solved_str, True, self.COLOR_TEXT)
        self.screen.blit(solved_text, (self.SCREEN_WIDTH - solved_text.get_width() - 20, 20))
        
        # Momentum
        if self.momentum_active:
            self._draw_flame((self.grid_x - 40, score_pos_y + 15))

    def _draw_flame(self, pos):
        x, y = pos
        points1 = [(x, y), (x + 20, y), (x + 10, y - 25)]
        points2 = [(x + 5, y), (x + 15, y), (x + 10, y - 15)]
        points3 = [(x + 8, y), (x + 12, y), (x + 10, y - 8)]
        pygame.draw.polygon(self.screen, self.COLOR_MOMENTUM_FLAME[2], points1)
        pygame.draw.polygon(self.screen, self.COLOR_MOMENTUM_FLAME[1], points2)
        pygame.draw.polygon(self.screen, self.COLOR_MOMENTUM_FLAME[0], points3)

    def _create_particles(self, pos, color, count, p_type):
        for _ in range(count):
            if p_type == 'solve':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 5)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(30, 60)
            elif p_type == 'land':
                angle = self.np_random.uniform(-math.pi * 0.8, -math.pi * 0.2)
                speed = self.np_random.uniform(0.5, 2)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(15, 30)
            
            self.particles.append({'x': pos[0], 'y': pos[1], 'vx': vx, 'vy': vy, 'life': life, 'max_life': life, 'color': color})

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = p['color']
                size = max(1, int(4 * (p['life'] / p['max_life'])))
                pygame.draw.circle(self.screen, color, (int(p['x']), int(p['y'])), size)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "equations_solved": self.equations_solved,
            "target_number": self.target_number,
            "momentum": self.momentum_active
        }

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
    # This block is for human play and debugging.
    # It will not be executed by the test environment.
    # To use, you might need to remove the dummy video driver setting.
    # e.g., comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    
    # For display, we need to unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a simple human-playable agent
    # 0=none, 1=up, 2=down, 3=left, 4=right
    action = np.array([0, 0, 0]) 

    # For displaying the rendered screen
    pygame.display.set_caption("Equation Grid")
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        env.clock.tick(30) # Run at 30 FPS

    env.close()