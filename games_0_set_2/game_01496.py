
# Generated: 2025-08-27T17:20:05.527985
# Source Brief: brief_01496.md
# Brief Index: 1496

        
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
        "Controls: Arrow keys to move the cursor. Space to plant a seed in the selected plot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate a thriving garden by strategically planting seeds. Plants will spread to adjacent empty plots each turn. Fill the garden before you run out of turns or seeds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.PLOT_SIZE = 36
        self.PLOT_PADDING = 4
        self.CELL_SIZE = self.PLOT_SIZE + self.PLOT_PADDING
        
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Colors
        self.COLOR_BG = (50, 30, 20)          # Dark soil background
        self.COLOR_PLOT = (101, 67, 33)       # Empty plot soil
        self.COLOR_PLANT = (76, 175, 80)      # Vibrant green
        self.COLOR_PLANT_CENTER = (129, 199, 132) # Lighter green
        self.COLOR_CURSOR = (255, 235, 59)    # Bright yellow
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER_BG = (0, 0, 0, 180)

        # Game parameters
        self.INITIAL_SEEDS = 5
        self.MAX_TURNS = 10
        self.GROWTH_RATE = 0.25 # Turns to reach full size = 1 / GROWTH_RATE

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_gameover = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_win_lose = pygame.font.SysFont("Arial", 32, bold=True)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.seeds = None
        self.turns_left = None
        self.score = None
        self.game_over = None
        self.win = None
        self.particles = []
        self.last_action_feedback = ""

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Grid: 0.0 = empty, > 0.0 = plant growth (0.0 to 1.0)
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.seeds = self.INITIAL_SEEDS
        self.turns_left = self.MAX_TURNS
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.particles = []
        self.last_action_feedback = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.last_action_feedback = ""

        # Unpack factorized action
        movement = action[0]
        plant_action = action[1] == 1

        # 1. Handle player input
        self._handle_movement(movement)
        if plant_action:
            self._handle_planting()

        # 2. Update game state (turn progression)
        newly_filled_plots = self._update_growth()
        self.turns_left -= 1
        
        # 3. Update particles
        self._update_particles()

        # 4. Calculate reward
        reward += newly_filled_plots  # +1 for each plot filled by spread

        # 5. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win:
                # Brief: +10 for filling garden, +100 for winning
                reward += 110
            else:
                reward -= 100
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

    def _handle_planting(self):
        cx, cy = self.cursor_pos
        if self.seeds > 0 and self.grid[cy][cx] == 0.0:
            self.seeds -= 1
            self.grid[cy][cx] = 0.01 # Start growth
            self.last_action_feedback = "Seed Planted!"
            # SFX: plant_seed.wav
            self._create_particles(cx, cy, self.COLOR_PLANT, 20)
        elif self.seeds <= 0:
            self.last_action_feedback = "No seeds left!"
        else: # Plot already occupied
            self.last_action_feedback = "Plot occupied!"


    def _update_growth(self):
        new_grid = self.grid.copy()
        plots_to_spawn = set()

        # First pass: identify where new plants will spawn
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] >= 1.0: # Fully grown
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and self.grid[nr, nc] == 0.0:
                            plots_to_spawn.add((nr, nc))
        
        # Second pass: grow existing plants
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if 0 < self.grid[r, c] < 1.0:
                    new_grid[r, c] = min(1.0, self.grid[r, c] + self.GROWTH_RATE)
        
        # Third pass: place new spawns
        for r, c in plots_to_spawn:
            if new_grid[r, c] == 0.0: # Ensure it wasn't planted this turn
                new_grid[r, c] = 0.01 # Start growth
                # SFX: plant_spread.wav
                self._create_particles(c, r, self.COLOR_PLANT_CENTER, 10)
        
        self.grid = new_grid
        return len(plots_to_spawn)

    def _check_termination(self):
        garden_is_full = np.all(self.grid > 0)

        if garden_is_full:
            self.game_over = True
            self.win = True
            return True
        
        # Check loss conditions only if not won
        if self.turns_left <= 0:
            self.game_over = True
            self.win = False
            return True

        # Check if we can still make a move
        can_plant = self.seeds > 0 and np.any(self.grid == 0.0)
        can_spread = np.any(self.grid > 0) # Plants can still spread
        if not can_plant and not can_spread:
             self.game_over = True
             self.win = False
             return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "seeds": self.seeds,
            "turns_left": self.turns_left,
            "cursor_pos": self.cursor_pos,
        }

    def _render_game(self):
        # Draw plots
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                screen_x = self.GRID_X_OFFSET + c * self.CELL_SIZE
                screen_y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
                
                # Draw base plot
                pygame.draw.rect(
                    self.screen,
                    self.COLOR_PLOT,
                    (screen_x, screen_y, self.PLOT_SIZE, self.PLOT_SIZE),
                    border_radius=4
                )
                
                # Draw plant if it exists
                growth = self.grid[r, c]
                if growth > 0:
                    center_x = screen_x + self.PLOT_SIZE // 2
                    center_y = screen_y + self.PLOT_SIZE // 2
                    radius = int(max(1, (self.PLOT_SIZE / 2) * math.sqrt(growth)))
                    
                    # Use gfxdraw for anti-aliased circles
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLANT)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLANT)
                    
                    # Inner, lighter circle for depth
                    if radius > 4:
                         pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius // 2, self.COLOR_PLANT_CENTER)
                         pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius // 2, self.COLOR_PLANT_CENTER)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Draw cursor
        cursor_x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE - self.PLOT_PADDING // 2
        cursor_y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE - self.PLOT_PADDING // 2
        cursor_size = self.PLOT_SIZE + self.PLOT_PADDING
        pygame.draw.rect(
            self.screen,
            self.COLOR_CURSOR,
            (cursor_x, cursor_y, cursor_size, cursor_size),
            width=3,
            border_radius=6
        )

    def _render_ui(self):
        # Helper to draw text with a shadow for readability
        def draw_text_shadow(text, font, color, pos):
            text_surf = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # UI elements
        draw_text_shadow(f"Seeds: {self.seeds}", self.font_ui, self.COLOR_TEXT, (20, 10))
        draw_text_shadow(f"Turns Left: {self.turns_left}", self.font_ui, self.COLOR_TEXT, (self.WIDTH - 180, 10))
        draw_text_shadow(f"Score: {self.score}", self.font_ui, self.COLOR_TEXT, (self.WIDTH // 2 - 50, 10))
        
        if self.last_action_feedback:
             feedback_surf = self.font_ui.render(self.last_action_feedback, True, self.COLOR_CURSOR)
             pos = (self.WIDTH // 2 - feedback_surf.get_width() // 2, self.HEIGHT - 30)
             self.screen.blit(feedback_surf, pos)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_GAMEOVER_BG)
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.font_gameover.render("GAME OVER", True, self.COLOR_TEXT)
        text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
        self.screen.blit(game_over_text, text_rect)
        
        result_text_str = "GARDEN COMPLETE!" if self.win else "TRY AGAIN"
        result_color = self.COLOR_PLANT if self.win else self.COLOR_CURSOR
        result_text = self.font_win_lose.render(result_text_str, True, result_color)
        result_rect = result_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
        self.screen.blit(result_text, result_rect)

    def _create_particles(self, grid_c, grid_r, color, count):
        px = self.GRID_X_OFFSET + grid_c * self.CELL_SIZE + self.PLOT_SIZE // 2
        py = self.GRID_Y_OFFSET + grid_r * self.CELL_SIZE + self.PLOT_SIZE // 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'color': color,
                'life': random.uniform(10, 20)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] -= 0.1
            p['life'] -= 1
            if p['radius'] <= 0 or p['life'] <= 0:
                self.particles.remove(p)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part requires a display. It will not run in a headless environment.
    # To run, you would need to modify the rendering to use pygame.display
    try:
        import sys
        # Check if we can use display
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Garden Cultivator")
    except pygame.error:
        print("No display available. Skipping manual play example.")
        sys.exit()

    obs, info = env.reset()
    done = False
    clock = pygame.time.Clock()

    print("\n--- Manual Game Start ---")
    print(env.game_description)
    print(env.user_guide)

    while not done:
        # Action defaults to NO-OP
        action = [0, 0, 0] 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
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
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit
                    done = True

        if any(action): # Only step if an action was taken
             obs, reward, terminated, truncated, info = env.step(action)
             done = terminated or truncated
             print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    print("Game Over. Final Score:", info['score'])
    env.close()