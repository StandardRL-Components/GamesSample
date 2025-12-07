
# Generated: 2025-08-28T02:45:58.830525
# Source Brief: brief_04554.md
# Brief Index: 4554

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist puzzle game where the player fills a 10x10 grid with color.
    The goal is to fill the entire grid within a limited number of moves.
    Players can earn extra moves by strategically creating 2x2 blocks of the same color.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to fill. Shift to change color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fill the canvas with color in as few moves as possible. Trigger 2x2 bonuses for extra moves and points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 36
        self.GRID_MARGIN = 2
        self.TOTAL_CELL_SIZE = self.CELL_SIZE + self.GRID_MARGIN
        self.GRID_WIDTH = self.GRID_SIZE * self.TOTAL_CELL_SIZE
        self.GRID_ORIGIN_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_ORIGIN_Y = (self.HEIGHT - self.GRID_WIDTH) // 2
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID_BG = (52, 73, 94)
        self.COLOR_CURSOR = (241, 196, 15)
        self.COLOR_CURSOR_OUTLINE = (243, 156, 18)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_BONUS_FLASH = (255, 255, 255)
        self.FILL_COLORS = [
            (26, 188, 156), (46, 204, 113), (52, 152, 219),
            (155, 89, 182), (230, 126, 34), (231, 76, 60),
            (241, 196, 15), (149, 165, 166)
        ]

        # --- Pygame Setup ---
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # --- State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.current_color_index = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.np_random = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = None
        self.bonus_flash_timer = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.current_color_index = 0
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.bonus_flash_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        self.steps += 1
        
        # --- Handle Actions ---
        # 1. Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # 2. Change Color (on rising edge of shift)
        if shift_held and not self.prev_shift_held:
            self.current_color_index = (self.current_color_index + 1) % len(self.FILL_COLORS)
            # SFX: color_swap.wav

        # 3. Fill Action (on rising edge of space)
        if space_held and not self.prev_space_held:
            x, y = self.cursor_pos
            if self.grid[y, x] == 0:  # Can only click empty cells
                self.moves_left -= 1
                
                color_id = self.current_color_index + 1 # 0 is empty
                
                # Perform flood fill
                num_filled = self._flood_fill(x, y, color_id)
                reward += num_filled
                # SFX: fill.wav
                self._spawn_particles(x, y, num_filled)

                # Check for bonus
                if self._check_bonus(x, y, color_id):
                    self.moves_left = min(self.MAX_MOVES, self.moves_left + 3)
                    reward += 5
                    bonus_filled = self._apply_bonus_fill(color_id)
                    reward += bonus_filled
                    self.bonus_flash_timer = 15 # frames
                    # SFX: bonus_trigger.wav
                
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self._update_particles()
        if self.bonus_flash_timer > 0:
            self.bonus_flash_timer -= 1

        self.score = np.count_nonzero(self.grid)

        # --- Check Termination ---
        grid_is_full = self.score == self.GRID_SIZE * self.GRID_SIZE
        out_of_moves = self.moves_left <= 0 and not grid_is_full
        
        if grid_is_full:
            reward += 50
            terminated = True
            self.game_over = True
            # SFX: win_jingle.wav
        elif out_of_moves:
            reward -= 50
            terminated = True
            self.game_over = True
            # SFX: lose_sound.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )
    
    def _flood_fill(self, x, y, color_id):
        if not (0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE) or self.grid[y, x] != 0:
            return 0
        
        q = deque([(x, y)])
        count = 0
        while q:
            cx, cy = q.popleft()
            if not (0 <= cx < self.GRID_SIZE and 0 <= cy < self.GRID_SIZE):
                continue
            if self.grid[cy, cx] == 0:
                self.grid[cy, cx] = color_id
                count += 1
                q.append((cx + 1, cy))
                q.append((cx - 1, cy))
                q.append((cx, cy + 1))
                q.append((cx, cy - 1))
        return count

    def _check_bonus(self, x, y, color_id):
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                is_bonus = True
                for i in range(2):
                    for j in range(2):
                        nx, ny = x + dx + i, y + dy + j
                        if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE) or self.grid[ny, nx] != color_id:
                            is_bonus = False
                            break
                    if not is_bonus:
                        break
                if is_bonus:
                    return True
        return False

    def _apply_bonus_fill(self, color_id):
        bonus_size = 3
        start_x = self.np_random.integers(0, self.GRID_SIZE - bonus_size + 1)
        start_y = self.np_random.integers(0, self.GRID_SIZE - bonus_size + 1)
        
        filled_count = 0
        for i in range(bonus_size):
            for j in range(bonus_size):
                if self.grid[start_y + j, start_x + i] == 0:
                    self.grid[start_y + j, start_x + i] = color_id
                    filled_count += 1
        if filled_count > 0:
            self._spawn_particles(start_x + 1, start_y + 1, filled_count)
        return filled_count
    
    def _spawn_particles(self, grid_x, grid_y, num_filled):
        center_x = self.GRID_ORIGIN_X + grid_x * self.TOTAL_CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_ORIGIN_Y + grid_y * self.TOTAL_CELL_SIZE + self.CELL_SIZE / 2
        
        num_particles = min(30, num_filled * 2)
        color = self.FILL_COLORS[self.current_color_index]

        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(3, 7)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'size': size,
                'life': life,
                'max_life': life,
                'color': color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Damping
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_area_rect = pygame.Rect(self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y, self.GRID_WIDTH, self.GRID_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_area_rect.inflate(self.GRID_MARGIN, self.GRID_MARGIN))
        
        # Draw filled cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_id = self.grid[y, x]
                if color_id > 0:
                    color = self.FILL_COLORS[color_id - 1]
                    rect = pygame.Rect(
                        self.GRID_ORIGIN_X + x * self.TOTAL_CELL_SIZE,
                        self.GRID_ORIGIN_Y + y * self.TOTAL_CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, color, rect)
        
        # Draw cursor
        cursor_x = self.GRID_ORIGIN_X + self.cursor_pos[0] * self.TOTAL_CELL_SIZE
        cursor_y = self.GRID_ORIGIN_Y + self.cursor_pos[1] * self.TOTAL_CELL_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR_OUTLINE, cursor_rect, border_radius=4)
        inner_rect = cursor_rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, inner_rect, border_radius=2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']),
                (color[0], color[1], color[2], alpha)
            )

        # Draw bonus flash
        if self.bonus_flash_timer > 0:
            alpha = int(255 * (self.bonus_flash_timer / 15))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_BONUS_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        # Current Color
        color_indicator_text = self.font_small.render("Color", True, self.COLOR_TEXT)
        self.screen.blit(color_indicator_text, (20, self.HEIGHT - 45))
        pygame.draw.rect(
            self.screen, self.FILL_COLORS[self.current_color_index],
            (20, self.HEIGHT - 25, 50, 20)
        )
        pygame.draw.rect(
            self.screen, self.COLOR_TEXT,
            (20, self.HEIGHT - 25, 50, 20), 1
        )

        # Game Over Text
        if self.game_over:
            grid_is_full = self.score == self.GRID_SIZE * self.GRID_SIZE
            message = "COMPLETE!" if grid_is_full else "GAME OVER"
            color = (100, 255, 100) if grid_is_full else (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = pygame.font.SysFont("monospace", 60, bold=True).render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
        }
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("✓ Running implementation validation...")
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

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It's a useful way to test and debug the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for interactive testing
    pygame.display.set_caption("Grid Fill Game")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    clock = pygame.time.Clock()
    
    print("--- Game Start ---")
    print(env.user_guide)
    
    while not terminated:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Since auto_advance is False, the environment only updates on step().
        # We call step() every frame and let the internal rising-edge detection
        # handle when actions are actually performed.
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # Render the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("\n--- RESETTING ---\n")
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    terminated = True

        clock.tick(30) # Limit to 30 FPS

    print("\nGame Over! Press R to restart or ESC to quit.")
    
    # Keep the window open to see the final screen
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                env.close()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # This part is just for the interactive demo, not part of the env itself
                main_loop = True
                print("\n--- RESTARTING ---\n")
                obs, info = env.reset()
                real_screen.blit(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), (0,0))
                pygame.display.flip()
                # A bit of a hack to break out and restart the main loop
                # A cleaner way would be to wrap the main loop in a function.
                # But for a simple demo, this is fine.
                exec(open(__file__).read())
                exit()
        clock.tick(30)