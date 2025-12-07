import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, dx, dy, gravity=0.1):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.dx = dx
        self.dy = dy
        self.gravity = gravity

    def update(self):
        self.life -= 1
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            radius = int(3 * (self.life / self.max_life))
            if radius > 0:
                color = (*self.color, alpha)
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selected block. "
        "Press Space to cycle which block is selected."
    )

    game_description = (
        "Recreate the target pattern by moving colored blocks on the grid. "
        "Correctly placed blocks glow green. Solve the puzzle before time runs out!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.BLOCK_SIZE = 32
        self.GRID_LINE_WIDTH = 1
        self.GRID_WIDTH = self.GRID_COLS * self.BLOCK_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.BLOCK_SIZE
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 3
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_BG = (40, 44, 52)
        self.COLOR_GRID_LINES = (60, 65, 75)
        self.COLOR_UI_PANEL = (33, 37, 45)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CORRECT = (70, 255, 120)
        self.COLOR_INCORRECT_OUTLINE = (255, 80, 80)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.BLOCK_COLORS = [
            (59, 179, 217),  # Blue
            (217, 59, 119),  # Magenta
            (216, 179, 89),  # Yellow
            (122, 217, 59),  # Lime
            (179, 59, 217),  # Purple
        ]
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 28)

        # Game state variables initialized in reset()
        self.rng = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.time_limit = 1000
        self.num_blocks = 5
        self.blocks = []
        self.target_pattern = []
        self.correct_mask = []
        self.selected_block_idx = 0
        self.last_space_held = False
        self.particles = []

        # self.reset() is called here to set up initial state, which is fine
        # but we defer it to the first external call to reset() as per Gym API best practices.
        # However, the original code had it, so let's keep it for compatibility with the test harness.
        # self.validate_implementation() also requires state to be initialized.
        self.reset()
        # self.validate_implementation() # This is a self-check, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rng is None: # Initialize RNG only once or on seed change
            self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.selected_block_idx = 0
        self.last_space_held = False
        self.particles = []
        
        self._generate_puzzle()
        self._update_correctness()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Input ---
        # Cycle selected block on space press (not hold)
        if space_held and not self.last_space_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % self.num_blocks
            sel_block = self.blocks[self.selected_block_idx]
            self._add_particles(
                self.GRID_X + sel_block['x'] * self.BLOCK_SIZE + self.BLOCK_SIZE // 2,
                self.GRID_Y + sel_block['y'] * self.BLOCK_SIZE + self.BLOCK_SIZE // 2,
                self.COLOR_SELECTOR, 10, 2.0
            )

        self.last_space_held = space_held
        
        # Move selected block
        if movement > 0:
            block = self.blocks[self.selected_block_idx]
            old_pos = (block['x'], block['y'])
            
            if movement == 1: block['y'] -= 1  # Up
            elif movement == 2: block['y'] += 1  # Down
            elif movement == 3: block['x'] -= 1  # Left
            elif movement == 4: block['x'] += 1  # Right
            
            # Grid wrapping
            block['x'] %= self.GRID_COLS
            block['y'] %= self.GRID_ROWS

            if (block['x'], block['y']) != old_pos:
                self._add_particles(
                    self.GRID_X + block['x'] * self.BLOCK_SIZE + self.BLOCK_SIZE // 2,
                    self.GRID_Y + block['y'] * self.BLOCK_SIZE + self.BLOCK_SIZE // 2,
                    block['color'], 15, 1.5
                )

        # --- Update State ---
        self.steps += 1
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.life > 0]
        
        self._update_correctness()
        is_solved = all(self.correct_mask)

        # --- Calculate Reward and Termination ---
        reward = 0
        terminated = False
        
        num_correct = sum(self.correct_mask)
        reward += num_correct * 0.1
        reward -= (self.num_blocks - num_correct) * 0.01

        if is_solved:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 15.0
        
        if self.steps >= self.time_limit:
            self.game_over = True
            terminated = True
            if not self.win:
                reward = -10.0

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_puzzle(self):
        possible_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.rng.shuffle(possible_positions)
        
        self.target_pattern = []
        for i in range(self.num_blocks):
            pos = possible_positions.pop()
            # FIX: self.rng.choice on a list of tuples returns a numpy array, which is unhashable.
            # Instead, we select an index and then get the color tuple to preserve its type.
            color_idx = self.rng.integers(len(self.BLOCK_COLORS))
            color = self.BLOCK_COLORS[color_idx]
            self.target_pattern.append({'x': pos[0], 'y': pos[1], 'color': color})
            
        # Create a shuffled version for the initial state
        self.blocks = [dict(b) for b in self.target_pattern]
        shuffle_steps = self.rng.integers(10, 25)
        for _ in range(shuffle_steps):
            block_to_move = self.rng.choice(self.blocks)
            direction = self.rng.integers(1, 5)
            if direction == 1: block_to_move['y'] -= 1
            elif direction == 2: block_to_move['y'] += 1
            elif direction == 3: block_to_move['x'] -= 1
            elif direction == 4: block_to_move['x'] += 1
            block_to_move['x'] %= self.GRID_COLS
            block_to_move['y'] %= self.GRID_ROWS

    def _update_correctness(self):
        # With the color now being a tuple, it is hashable and can be used in a set.
        target_set = {(b['x'], b['y'], b['color']) for b in self.target_pattern}
        self.correct_mask = [
            (b['x'], b['y'], b['color']) in target_set for b in self.blocks
        ]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game_elements()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_limit - self.steps,
            "correct_blocks": sum(self.correct_mask),
        }
        
    def _render_game_elements(self):
        # Draw main grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        # Draw blocks and correctness indicators
        for i, block in enumerate(self.blocks):
            is_correct = self.correct_mask[i]
            rect = pygame.Rect(
                self.GRID_X + block['x'] * self.BLOCK_SIZE,
                self.GRID_Y + block['y'] * self.BLOCK_SIZE,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            
            # Draw block with slight inset
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, block['color'], inner_rect, border_radius=4)

            # Visual feedback for correctness
            if is_correct:
                pygame.draw.rect(self.screen, self.COLOR_CORRECT, rect, 2, border_radius=6)
            else:
                pygame.draw.rect(self.screen, self.COLOR_INCORRECT_OUTLINE, rect, 1, border_radius=6)

        # Draw grid lines over blocks
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), self.GRID_LINE_WIDTH)

        # Draw pulsing selector
        if not self.game_over:
            selected_block = self.blocks[self.selected_block_idx]
            pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
            width = 2 + int(pulse * 2)
            alpha = 150 + int(pulse * 105)
            color = (*self.COLOR_SELECTOR, alpha)
            
            sel_rect = pygame.Rect(
                self.GRID_X + selected_block['x'] * self.BLOCK_SIZE,
                self.GRID_Y + selected_block['y'] * self.BLOCK_SIZE,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), width, border_radius=6)
            self.screen.blit(s, sel_rect.topleft)

    def _render_ui(self):
        ui_panel_x = self.GRID_X + self.GRID_WIDTH + 20
        ui_panel_width = self.SCREEN_WIDTH - ui_panel_x - 20
        ui_panel_rect = pygame.Rect(ui_panel_x, self.GRID_Y, ui_panel_width, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_PANEL, ui_panel_rect, border_radius=8)

        # Title: Target
        title_surf = self.font_title.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (ui_panel_rect.centerx - title_surf.get_width() // 2, ui_panel_rect.top + 15))
        
        # Target Pattern Display
        target_grid_size = 16
        target_grid_width = self.GRID_COLS * target_grid_size
        target_grid_height = self.GRID_ROWS * target_grid_size
        target_x_start = ui_panel_rect.centerx - target_grid_width // 2
        target_y_start = ui_panel_rect.top + 50
        
        for block in self.target_pattern:
            rect = pygame.Rect(
                target_x_start + block['x'] * target_grid_size,
                target_y_start + block['y'] * target_grid_size,
                target_grid_size, target_grid_size
            )
            pygame.draw.rect(self.screen, block['color'], rect.inflate(-2,-2), border_radius=2)

        # Score and Time Display
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.GRID_X, self.GRID_Y - 40))
        
        time_left = max(0, self.time_limit - self.steps)
        time_text = f"TIME: {time_left}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 20, self.GRID_Y - 40))

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)
            
    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(overlay, (0, 0))
        
        message = "PUZZLE SOLVED!" if self.win else "TIME UP!"
        color = self.COLOR_CORRECT if self.win else self.COLOR_INCORRECT_OUTLINE
        
        text_surf = self.font_main.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)

    def _add_particles(self, x, y, color, count, max_speed):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(0.5, max_speed)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            life = self.rng.integers(15, 31) # Use self.rng and integers for reproducibility
            self.particles.append(Particle(x, y, color, life, dx, dy))
            
    def render(self):
        return self._get_observation()

if __name__ == "__main__":
    # To play the game manually
    # Note: The original code included a `validate_implementation` method which is good for testing
    # but removed here to keep the final output clean.
    
    # We need to unset the dummy video driver to see the display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Pixel Pattern Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space_press = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # This logic is slightly different from the environment's internal logic
        # which checks for press-and-release. For manual play, holding is fine.
        if keys[pygame.K_SPACE]:
            space_press = 1
            
        action = [movement, space_press, 0] 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
        
        clock.tick(30) 
        
    pygame.quit()