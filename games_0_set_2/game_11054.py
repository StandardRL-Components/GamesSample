import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    A visually-focused Gymnasium environment for a quantum Sudoku puzzle game.

    The player interacts with a Sudoku-like grid, using quantum portals to reveal
    or swap numbers. The goal is to solve the puzzle. The environment is designed
    for visual quality and satisfying gameplay feel, making it suitable for both
    human players and reinforcement learning agents.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0:None, 1:Up, 2:Down, 3:Left, 4:Right)
    - action[1]: Space Button (0:Released, 1:Held) - Primary action
    - action[2]: Shift Button (0:Released, 1:Held) - Mode switch

    Gameplay Modes:
    - Normal Mode (Shift Released): Movement keys move the cursor. Space key places/activates portals.
    - Input Mode (Shift Held): Movement keys (Up/Down) cycle the candidate number. Space key places the number.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Solve a Sudoku-like puzzle on a quantum grid. Use portals to reveal or swap numbers to complete the board."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to place portals. Hold shift to enter input mode, "
        "where ↑/↓ cycles numbers and space places them."
    )
    auto_advance = True

    # --- Colors and Style ---
    COLOR_BG = (10, 15, 25)
    COLOR_GRID = (50, 70, 100)
    COLOR_CURSOR = (0, 150, 255)
    COLOR_PORTAL_1 = (0, 255, 255)
    COLOR_PORTAL_2 = (255, 0, 255)
    COLOR_FIXED_NUM = (200, 200, 200)
    COLOR_PLAYER_CORRECT = (0, 255, 100)
    COLOR_PLAYER_INCORRECT = (255, 80, 80)
    COLOR_CANDIDATE_NUM = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # --- Game Constants ---
    MAX_STEPS = 1000
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_AREA_SIZE = 360 # Square area for the grid
    
    class Effect:
        """A simple class for managing temporary visual effects like text pop-ups."""
        def __init__(self, pos, text, color, lifetime=30, font=None):
            self.pos = list(pos)
            self.text = text
            self.color = color
            self.lifetime = lifetime
            self.max_lifetime = lifetime
            self.font = font

        def update(self):
            self.lifetime -= 1
            self.pos[1] -= 0.5 # Move upwards
            return self.lifetime > 0

        def draw(self, surface):
            if not self.font: return
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            text_surf = self.font.render(self.text, True, self.color)
            text_surf.set_alpha(max(0, alpha))
            surface.blit(text_surf, (int(self.pos[0]), int(self.pos[1])))

    class Particle:
        """A simple class for particle effects, used for portals."""
        def __init__(self, pos, vel, radius, color, lifetime):
            self.pos = list(pos)
            self.vel = list(vel)
            self.radius = radius
            self.color = color
            self.lifetime = lifetime
            self.max_lifetime = lifetime
        
        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.lifetime -= 1
            self.radius *= 0.98
            return self.lifetime > 0

        def draw(self, surface):
            alpha = int(255 * (self.lifetime / self.max_lifetime)**2)
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), color)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18)
        self.font_grid_base = pygame.font.SysFont("Segoe UI", 32)
        self.font_effect = pygame.font.SysFont("Arial Black", 20)
        
        # Game progression state (persists across resets)
        self.level = 1
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid_size = 0
        self.initial_reveal_pct = 0.0
        self.solution_grid = []
        self.player_grid = []
        self.fixed_grid = []
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0.0, 0.0]
        self.prev_space_held = False
        self.candidate_number = 1
        self.portal_1_pos = None
        self.portal_2_pos = None
        self.portal_activation_timer = 0
        self.particles = []
        self.effects = []
        self.shift_held = False
        
        self.reset()
        self.validate_implementation()

    def _generate_puzzle(self, size, reveal_pct):
        """Generates a solvable Latin Square (Sudoku-like) puzzle."""
        # 1. Create a solved grid (Latin Square)
        base = list(range(1, size + 1))
        solution = [base[i:] + base[:i] for i in range(size)]
        
        # 2. Shuffle it to create variety
        self.np_random.shuffle(solution) # Shuffle rows
        solution = np.array(solution).T.tolist()
        self.np_random.shuffle(solution) # Shuffle columns
        
        shuffled_nums = list(range(1, size + 1))
        self.np_random.shuffle(shuffled_nums)
        num_map = {i + 1: shuffled_nums[i] for i in range(size)}
        solution = [[num_map[cell] for cell in row] for row in solution]

        # 3. Create player grid by "poking holes"
        player_grid = [row[:] for row in solution]
        fixed_grid = [[True] * size for _ in range(size)]
        
        cells = [(r, c) for r in range(size) for c in range(size)]
        self.np_random.shuffle(cells)
        
        num_to_hide = int(size * size * (1 - reveal_pct))
        for r, c in cells[:num_to_hide]:
            player_grid[r][c] = 0
            fixed_grid[r][c] = False
            
        return solution, player_grid, fixed_grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Update progression based on level
        self.grid_size = 9 # Simplified for consistency, was: 9 + (self.level - 1) // 3
        self.initial_reveal_pct = max(0.15, 0.5 - (self.level - 1) * 0.02)
        
        self.solution_grid, self.player_grid, self.fixed_grid = self._generate_puzzle(self.grid_size, self.initial_reveal_pct)
        
        self.cursor_pos = [self.grid_size // 2, self.grid_size // 2]
        self.visual_cursor_pos = [float(c) for c in self.cursor_pos]
        self.prev_space_held = False
        self.candidate_number = 1
        self.portal_1_pos = None
        self.portal_2_pos = None
        self.portal_activation_timer = 0
        self.particles = []
        self.effects = []
        self.shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.shift_held = shift_held
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        # --- Handle Input ---
        if self.shift_held: # Input Mode
            if movement == 1: # Up
                self.candidate_number = (self.candidate_number % self.grid_size) + 1
            elif movement == 2: # Down
                self.candidate_number = self.grid_size if self.candidate_number == 1 else self.candidate_number - 1
            
            if space_pressed:
                cx, cy = self.cursor_pos
                if not self.fixed_grid[cy][cx]:
                    # sfx: number_place
                    old_val = self.player_grid[cy][cx]
                    if old_val != self.candidate_number:
                        is_correct = self.candidate_number == self.solution_grid[cy][cx]
                        reward += 1.0 if is_correct else -1.0
                        self.score += 1 if is_correct else -1
                        
                        effect_color = self.COLOR_PLAYER_CORRECT if is_correct else self.COLOR_PLAYER_INCORRECT
                        effect_text = "+1" if is_correct else "-1"
                        cell_center = self._get_cell_center(cx, cy)
                        self.effects.append(self.Effect(cell_center, effect_text, effect_color, font=self.font_effect))
                    
                    self.player_grid[cy][cx] = self.candidate_number

        else: # Move/Portal Mode
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.grid_size - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.grid_size - 1, self.cursor_pos[0] + 1)
            
            if space_pressed:
                if self.portal_1_pos is None:
                    # sfx: portal_open_1
                    self.portal_1_pos = list(self.cursor_pos)
                elif self.portal_2_pos is None and self.cursor_pos != self.portal_1_pos:
                    # sfx: portal_open_2
                    self.portal_2_pos = list(self.cursor_pos)
                    self.portal_activation_timer = 60 # Linger for visual effect

        # --- Update Game Logic ---
        self.steps += 1
        self._update_animations()
        
        if self.portal_activation_timer > 0:
            self.portal_activation_timer -= 1
            if self.portal_activation_timer == 30: # Trigger effect mid-animation
                self._trigger_portal_effect()
            if self.portal_activation_timer == 0:
                self.portal_1_pos = self.portal_2_pos = None
        
        # --- Check Termination ---
        terminated = False
        is_solved = all(self.player_grid[r][c] == self.solution_grid[r][c] for r in range(self.grid_size) for c in range(self.grid_size))
        
        if is_solved:
            # sfx: level_complete
            reward += 100.0
            self.score += 100
            terminated = True
            self.game_over = True
            self.level += 1
        elif self.steps >= self.MAX_STEPS:
            reward -= 10.0
            self.score -= 10
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _trigger_portal_effect(self):
        # sfx: portal_activate
        p1x, p1y = self.portal_1_pos
        p2x, p2y = self.portal_2_pos
        
        # 50/50 chance to reveal or swap
        if self.np_random.random() < 0.5: # Reveal
            portal_to_reveal = self.portal_1_pos if self.np_random.random() < 0.5 else self.portal_2_pos
            rx, ry = portal_to_reveal
            if self.player_grid[ry][rx] == 0: # If number is hidden
                self.player_grid[ry][rx] = self.solution_grid[ry][rx]
                self.fixed_grid[ry][rx] = True # Revealed numbers become fixed
                self.score += 5
                cell_center = self._get_cell_center(rx, ry)
                self.effects.append(self.Effect(cell_center, "REVEAL", self.COLOR_PORTAL_1, font=self.font_effect))

        else: # Swap
            if not self.fixed_grid[p1y][p1x] and not self.fixed_grid[p2y][p2x]:
                self.player_grid[p1y][p1x], self.player_grid[p2y][p2x] = \
                    self.player_grid[p2y][p2x], self.player_grid[p1y][p1x]
                cell_center = self._get_cell_center(p1x, p1y)
                self.effects.append(self.Effect(cell_center, "SWAP", self.COLOR_PORTAL_2, font=self.font_effect))

    def _update_animations(self):
        # Smooth cursor movement
        lerp_rate = 0.3
        self.visual_cursor_pos[0] += (self.cursor_pos[0] - self.visual_cursor_pos[0]) * lerp_rate
        self.visual_cursor_pos[1] += (self.cursor_pos[1] - self.visual_cursor_pos[1]) * lerp_rate

        # Update particles
        if self.portal_1_pos:
            center = self._get_cell_center(*self.portal_1_pos)
            for _ in range(2): self._emit_particle(center, self.COLOR_PORTAL_1)
        if self.portal_2_pos:
            center = self._get_cell_center(*self.portal_2_pos)
            for _ in range(2): self._emit_particle(center, self.COLOR_PORTAL_2)
        
        self.particles = [p for p in self.particles if p.update()]
        self.effects = [e for e in self.effects if e.update()]

    def _emit_particle(self, pos, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, 1.5)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        radius = self.np_random.uniform(2, 5)
        lifetime = self.np_random.integers(20, 40)
        self.particles.append(self.Particle(pos, vel, radius, color, lifetime))

    def _get_cell_metrics(self):
        cell_size = self.GRID_AREA_SIZE / self.grid_size
        offset_x = (self.SCREEN_WIDTH - self.GRID_AREA_SIZE) / 2
        offset_y = (self.SCREEN_HEIGHT - self.GRID_AREA_SIZE) / 2
        return cell_size, offset_x, offset_y

    def _get_cell_center(self, x, y):
        cell_size, offset_x, offset_y = self._get_cell_metrics()
        return (offset_x + (x + 0.5) * cell_size, offset_y + (y + 0.5) * cell_size)

    def _render_game(self):
        cell_size, offset_x, offset_y = self._get_cell_metrics()
        
        # Render particles (background)
        for p in self.particles: p.draw(self.screen)
        
        # Render portals
        self._draw_portal(self.portal_1_pos, self.COLOR_PORTAL_1)
        self._draw_portal(self.portal_2_pos, self.COLOR_PORTAL_2)
        if self.portal_1_pos and self.portal_2_pos and self.portal_activation_timer > 0:
            p1_center = self._get_cell_center(*self.portal_1_pos)
            p2_center = self._get_cell_center(*self.portal_2_pos)
            
            progress = (60 - self.portal_activation_timer) / 30.0
            if 0 < progress <= 1.0:
                mid_point = (p1_center[0] * (1-progress) + p2_center[0] * progress,
                             p1_center[1] * (1-progress) + p2_center[1] * progress)
                pygame.draw.circle(self.screen, (255,255,255), mid_point, 10, 2)

        # Render cursor
        cursor_rect = pygame.Rect(
            offset_x + self.visual_cursor_pos[0] * cell_size,
            offset_y + self.visual_cursor_pos[1] * cell_size,
            cell_size, cell_size
        )
        cursor_surf = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        cursor_color = (255, 255, 0) if self.shift_held else self.COLOR_CURSOR
        pygame.draw.rect(cursor_surf, cursor_color + (80,), (0, 0, cell_size, cell_size))
        pygame.draw.rect(cursor_surf, cursor_color, (0, 0, cell_size, cell_size), 3, border_radius=4)
        self.screen.blit(cursor_surf, cursor_rect.topleft)
        
        # Render grid and numbers
        font_size = int(cell_size * 0.6)
        font_grid = pygame.font.SysFont("Segoe UI", font_size, bold=True)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(offset_x + c * cell_size, offset_y + r * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
                
                num = self.player_grid[r][c]
                if num != 0:
                    color = self.COLOR_FIXED_NUM
                    if not self.fixed_grid[r][c]:
                        color = self.COLOR_PLAYER_CORRECT if num == self.solution_grid[r][c] else self.COLOR_PLAYER_INCORRECT
                    
                    text_surf = font_grid.render(str(num), True, color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)

        # Render candidate number in input mode
        if self.shift_held:
            cx, cy = self.cursor_pos
            rect = pygame.Rect(offset_x + cx * cell_size, offset_y + cy * cell_size, cell_size, cell_size)
            text_surf = font_grid.render(str(self.candidate_number), True, self.COLOR_CANDIDATE_NUM)
            text_surf.set_alpha(150)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
            
        # Render effects
        for e in self.effects: e.draw(self.screen)

    def _draw_portal(self, pos, color):
        if pos is None: return
        center = self._get_cell_center(*pos)
        cell_size, _, _ = self._get_cell_metrics()
        max_radius = cell_size * 0.45
        
        # Animate portal opening/closing
        t = (pygame.time.get_ticks() % 1000) / 1000.0
        
        for i in range(5):
            radius = max_radius * (0.6 + 0.4 * math.sin(t * 2 * math.pi + i * 0.5))
            alpha = 50 + i * 20
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), int(radius), color + (alpha,))

    def _render_ui(self):
        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))
        
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))
        
        steps_text = self.font_ui.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

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
            "level": self.level,
            "cursor_pos": list(self.cursor_pos),
            "is_solved": all(self.player_grid[r][c] == self.solution_grid[r][c] for r in range(self.grid_size) for c in range(self.grid_size))
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
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
    # --- Manual Play Example ---
    # This block will not run in the headless test environment.
    # It requires a display to be available.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a persistent screen for rendering if playing manually
    manual_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Sudoku")
    
    # Control mapping for human player
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_w: 1,
        pygame.K_DOWN: 2,
        pygame.K_s: 2,
        pygame.K_LEFT: 3,
        pygame.K_a: 3,
        pygame.K_RIGHT: 4,
        pygame.K_d: 4,
    }

    while not done:
        # Default action is "do nothing"
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()

        # Check for movement keys only once per frame
        for key, move_val in key_to_movement.items():
            if keys[key]:
                movement_action = move_val
                break 

        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
        
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Render the observation to the manual display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        manual_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # The environment internally uses a 30FPS assumption for animations.
        # Running the manual loop at 30FPS will make it look as intended.
        env.clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()