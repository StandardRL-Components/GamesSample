import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap."
    )

    # Must be a user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. "
        "Collect 100 gems within 20 moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.GEM_SIZE = 40
        self.GRID_X = (self.WIDTH - self.GRID_COLS * self.GEM_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_ROWS * self.GEM_SIZE) // 2
        self.NUM_GEM_TYPES = 6
        self.MAX_MOVES = 20
        self.WIN_GEMS = 100
        self.MAX_STEPS = 1000

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 60, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.GRID_BG_COLOR = (25, 35, 55)
        self.GRID_LINE_COLOR = (40, 55, 80)
        self.CURSOR_COLOR = (255, 255, 255)
        self.TEXT_COLOR = (220, 230, 255)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 160, 80),  # Orange
        ]
        
        # State variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.moves_left = 0
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.particles = []
        self.last_action_was_press = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.game_won = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_pos = None
        self.particles = []
        self.last_action_was_press = False

        self._create_stable_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0
        terminated = False
        truncated = False

        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS

        # Selection logic (only on button press, not hold)
        is_press = space_held and not self.last_action_was_press
        self.last_action_was_press = bool(space_held)

        if is_press:
            if self.selected_pos is None:
                # Select a gem
                self.selected_pos = list(self.cursor_pos)
            else:
                # Attempt a swap
                if self._is_adjacent(self.selected_pos, self.cursor_pos):
                    reward += self._attempt_swap(self.selected_pos, self.cursor_pos)
                elif self.selected_pos == list(self.cursor_pos):
                     # Deselect if clicking the same gem
                     self.selected_pos = None
                else:
                    # Select a new gem if clicking a non-adjacent one
                    self.selected_pos = list(self.cursor_pos)
        
        # --- Check Termination ---
        if self.gems_collected >= self.WIN_GEMS:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            self.game_won = False
            terminated = True
            reward -= 10
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _attempt_swap(self, pos1, pos2):
        # Swap gems
        self.grid[pos1[1], pos1[0]], self.grid[pos2[1], pos2[0]] = \
            self.grid[pos2[1], pos2[0]], self.grid[pos1[1], pos1[0]]
        
        self.moves_left -= 1
        
        all_matches = self._find_all_matches()

        if not all_matches:
            # Invalid swap, swap back
            self.grid[pos1[1], pos1[0]], self.grid[pos2[1], pos2[0]] = \
                self.grid[pos2[1], pos2[0]], self.grid[pos1[1], pos1[0]]
            self.selected_pos = None
            return 0
        
        # Valid swap, process matches
        self.selected_pos = None
        total_reward = 0
        
        chain_reaction_count = 0
        while all_matches:
            chain_reaction_count += 1
            
            # Get unique set of matched gems
            gems_to_remove = set()
            for match in all_matches:
                for pos in match:
                    gems_to_remove.add(pos)
            
            # Calculate reward
            num_cleared = len(gems_to_remove)
            total_reward += num_cleared  # +1 per gem
            self.gems_collected += num_cleared
            self.score += num_cleared * chain_reaction_count # Bonus for chains

            for match in all_matches:
                if len(match) == 4:
                    total_reward += 5
                    self.score += 5
                elif len(match) >= 5:
                    total_reward += 10
                    self.score += 10

            # Remove gems and create particles
            for r, c in gems_to_remove:
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = -1

            # Apply gravity and refill
            self._apply_gravity()
            self._refill_top_rows()

            # Check for new matches
            all_matches = self._find_all_matches()

        return total_reward

    def _find_all_matches(self):
        all_matches = []
        # Horizontal matches
        for r in range(self.GRID_ROWS):
            c = 0
            while c < self.GRID_COLS - 2:
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    match = [(r, c), (r, c+1), (r, c+2)]
                    i = c + 3
                    while i < self.GRID_COLS and self.grid[r, i] == self.grid[r, c]:
                        match.append((r, i))
                        i += 1
                    all_matches.append(match)
                    c = i
                else:
                    c += 1
        
        # Vertical matches
        for c in range(self.GRID_COLS):
            r = 0
            while r < self.GRID_ROWS - 2:
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    match = [(r, c), (r+1, c), (r+2, c)]
                    i = r + 3
                    while i < self.GRID_ROWS and self.grid[i, c] == self.grid[r, c]:
                        match.append((i, c))
                        i += 1
                    all_matches.append(match)
                    r = i
                else:
                    r += 1
        return all_matches

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _refill_top_rows(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _create_stable_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))
        while self._find_all_matches():
            all_matches = self._find_all_matches()
            gems_to_remove = set()
            for match in all_matches:
                for pos in match:
                    gems_to_remove.add(pos)
            for r, c in gems_to_remove:
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        return self._get_observation()

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_COLS * self.GEM_SIZE, self.GRID_ROWS * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.GRID_BG_COLOR, grid_rect, border_radius=8)
        
        self._update_and_draw_particles()
        self._draw_gems()
        self._draw_cursor_and_selection()

        # Draw grid lines on top
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.GEM_SIZE
            pygame.draw.line(self.screen, self.GRID_LINE_COLOR, (self.GRID_X, y), (self.GRID_X + self.GRID_COLS * self.GEM_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.GEM_SIZE
            pygame.draw.line(self.screen, self.GRID_LINE_COLOR, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_ROWS * self.GEM_SIZE))

    def _draw_gems(self):
        radius = self.GEM_SIZE // 2 - 5
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    center_x = self.GRID_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
                    center_y = self.GRID_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
                    color = self.GEM_COLORS[gem_type]
                    
                    # Draw gem body
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    # FIX: The color argument must be a tuple, not a generator.
                    outline_color = tuple(min(255, val + 50) for val in color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, outline_color)
                    
                    # Draw highlight
                    highlight_pos = (center_x - radius // 2, center_y - radius // 2)
                    pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 4, (255,255,255,100))

    def _draw_cursor_and_selection(self):
        # Draw selection highlight
        if self.selected_pos is not None:
            c, r = self.selected_pos
            x = self.GRID_X + c * self.GEM_SIZE
            y = self.GRID_Y + r * self.GEM_SIZE
            
            # Pulsing glow effect
            pulse = (math.sin(self.steps * 0.3) + 1) / 2  # Varies between 0 and 1
            size = int(self.GEM_SIZE * (1.1 + pulse * 0.2))
            offset = (size - self.GEM_SIZE) // 2
            
            glow_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (255, 255, 100, 50 + int(pulse * 50)), glow_surf.get_rect(), border_radius=12)
            self.screen.blit(glow_surf, (x - offset, y - offset))

        # Draw cursor
        c, r = self.cursor_pos
        rect = pygame.Rect(self.GRID_X + c * self.GEM_SIZE, self.GRID_Y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.CURSOR_COLOR, rect, width=3, border_radius=6)

    def _render_ui(self):
        # Gems Collected
        gem_text = f"GEMS: {self.gems_collected}/{self.WIN_GEMS}"
        gem_surf = self.font_ui.render(gem_text, True, self.TEXT_COLOR)
        self.screen.blit(gem_surf, (20, 10))

        # Moves Left
        moves_text = f"MOVES: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.TEXT_COLOR)
        self.screen.blit(moves_surf, (self.WIDTH - moves_surf.get_width() - 20, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.TEXT_COLOR)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, self.HEIGHT - 35))
    
    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = "YOU WIN!" if self.game_won else "GAME OVER"
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        
        text_surf = self.font_big.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _create_particles(self, c, r, gem_type):
        center_x = self.GRID_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.GRID_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        if gem_type < 0 or gem_type >= len(self.GEM_COLORS):
            return
        color = self.GEM_COLORS[gem_type]
        for _ in range(15): # Create 15 particles
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            pos = [center_x, center_y]
            life = random.randint(15, 30) # Lifespan in steps
            self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Damping
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
                
                # Draw particle
                alpha = int(255 * (p['life'] / p['max_life']))
                # Pygame rects don't handle alpha in their color argument, need to use a surface
                size = int(6 * (p['life'] / p['max_life']))
                if size > 0:
                    particle_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                    particle_surf.fill((*p['color'], alpha))
                    self.screen.blit(particle_surf, (p['pos'][0] - size//2, p['pos'][1] - size//2))
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "moves_left": self.moves_left,
        }
    
    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    import gymnasium.utils.play

    env = GameEnv(render_mode="rgb_array")
    
    # Test the environment with random actions
    print("Testing with random actions...")
    obs, info = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()
    print("Random action test complete.")
    env.close()

    # Interactive play
    print("\nStarting interactive game. Close the window to exit.")
    print(f"Game: {GameEnv.game_description}")
    print(f"Controls: {GameEnv.user_guide}")

    # Mapping from keyboard keys to MultiDiscrete actions
    # This is a simplified mapping; a more complex one could handle held keys.
    key_to_action = {
        "w": np.array([1, 0, 0]),
        "s": np.array([2, 0, 0]),
        "a": np.array([3, 0, 0]),
        "d": np.array([4, 0, 0]),
        " ": np.array([0, 1, 0]),
    }

    try:
        # The play utility from Gymnasium is great for interactive testing.
        gymnasium.utils.play.play(
            GameEnv(render_mode="rgb_array"),
            keys_to_action=key_to_action,
            fps=30,
            zoom=2
        )
    except Exception as e:
        print(f"Error during interactive play: {e}")
        print("Please ensure you have pygame installed (`pip install pygame`)")