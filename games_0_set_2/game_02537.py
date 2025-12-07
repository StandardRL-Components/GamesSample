
# Generated: 2025-08-28T05:09:47.155579
# Source Brief: brief_02537.md
# Brief Index: 2537

        
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

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a fruit. "
        "Select an adjacent fruit to swap. Shift to deselect."
    )

    game_description = (
        "Match 3 or more fruits to clear them from the board. "
        "Clear the whole board to win, but watch your move count!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.FRUIT_SIZE = 40
        self.GRID_LINE_WIDTH = 2
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # Centering the grid
        self.GRID_AREA_WIDTH = self.GRID_WIDTH * self.FRUIT_SIZE
        self.GRID_AREA_HEIGHT = self.GRID_HEIGHT * self.FRUIT_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_HEIGHT) // 2

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.FRUIT_COLORS = [
            (220, 50, 50),   # 1: Apple (Red)
            (50, 220, 50),   # 2: Lime (Green)
            (80, 80, 255),   # 3: Blueberry (Blue)
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.particles = []
        self.space_was_held = False # For press detection

        # Initialize state
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.particles = []
        self.space_was_held = False
        
        self._generate_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.particles.clear()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.space_was_held
        self.space_was_held = space_held

        # --- Action Handling ---
        # 1. Deselection with Shift
        if shift_held and self.selected_pos:
            self.selected_pos = None
            # sfx: deselect sound

        # 2. Cursor Movement
        elif movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT
            # sfx: cursor move tick

        # 3. Selection / Swap with Space
        elif space_pressed:
            if not self.selected_pos:
                # First selection
                self.selected_pos = list(self.cursor_pos)
                # sfx: select sound
            else:
                # Second selection (attempt swap)
                x1, y1 = self.selected_pos
                x2, y2 = self.cursor_pos
                
                # Check for adjacency
                if abs(x1 - x2) + abs(y1 - y2) == 1:
                    self.moves_left -= 1
                    reward -= 0.1 # Small penalty for any move attempt
                    
                    self._swap_fruits(x1, y1, x2, y2)
                    # sfx: swap attempt sound
                    
                    total_cleared, combo_bonus = self._handle_cascades()

                    if total_cleared > 0:
                        # Successful match
                        reward += total_cleared # +1 per fruit
                        reward += combo_bonus
                        self.score += total_cleared + combo_bonus
                        # sfx: successful match sound
                    else:
                        # Failed swap, swap back
                        self._swap_fruits(x1, y1, x2, y2) # No visual change
                        # sfx: invalid swap sound
                    
                    self.selected_pos = None # Deselect after any attempt
                else:
                    # Not adjacent, just move selection
                    self.selected_pos = list(self.cursor_pos)
                    # sfx: select sound

        # --- Termination Check ---
        board_cleared = np.all(self.grid == 0)
        out_of_moves = self.moves_left <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        
        terminated = board_cleared or out_of_moves or max_steps_reached
        if terminated and not self.game_over:
            self.game_over = True
            if board_cleared:
                reward += 50
                self.score += 50
                # sfx: win fanfare
            else:
                reward -= 50
                # sfx: lose sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Game Logic Helpers ---

    def _generate_initial_grid(self):
        while True:
            self.grid = self.np_random.integers(1, len(self.FRUIT_COLORS) + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            
            # Ensure no matches on start
            while self._find_matches():
                self._handle_cascades(generate_particles=False)

            # Ensure at least one move is possible
            if self._has_possible_moves():
                break

    def _swap_fruits(self, x1, y1, x2, y2):
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[c, r] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[c, r] == self.grid[c+1, r] == self.grid[c+2, r]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[c, r] == self.grid[c, r+1] == self.grid[c, r+2]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return list(matches)

    def _has_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                val = self.grid[c, r]
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[c, r], self.grid[c+1, r] = self.grid[c+1, r], self.grid[c, r]
                    if self._find_matches():
                        self.grid[c, r], self.grid[c+1, r] = self.grid[c+1, r], self.grid[c, r]
                        return True
                    self.grid[c, r], self.grid[c+1, r] = self.grid[c+1, r], self.grid[c, r]
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[c, r], self.grid[c, r+1] = self.grid[c, r+1], self.grid[c, r]
                    if self._find_matches():
                        self.grid[c, r], self.grid[c, r+1] = self.grid[c, r+1], self.grid[c, r]
                        return True
                    self.grid[c, r], self.grid[c, r+1] = self.grid[c, r+1], self.grid[c, r]
        return False

    def _handle_cascades(self, generate_particles=True):
        total_cleared = 0
        combo_bonus = 0
        combo_multiplier = 1

        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            num_cleared = len(matches)
            if num_cleared >= 4:
                combo_bonus += 5 * combo_multiplier
            total_cleared += num_cleared
            
            # Clear matched fruits and create particles
            for c, r in matches:
                fruit_type = self.grid[c, r]
                if generate_particles and fruit_type > 0:
                    self._create_particles(c, r, self.FRUIT_COLORS[fruit_type-1])
                self.grid[c, r] = 0
            
            # Apply gravity
            for c in range(self.GRID_WIDTH):
                empty_row = self.GRID_HEIGHT - 1
                for r in range(self.GRID_HEIGHT - 1, -1, -1):
                    if self.grid[c, r] != 0:
                        self.grid[c, empty_row], self.grid[c, r] = self.grid[c, r], self.grid[c, empty_row]
                        empty_row -= 1
            
            # Refill board
            for c in range(self.GRID_WIDTH):
                for r in range(self.GRID_HEIGHT):
                    if self.grid[c, r] == 0:
                        self.grid[c, r] = self.np_random.integers(1, len(self.FRUIT_COLORS) + 1)
            
            combo_multiplier += 1
            # sfx: cascade sound
        
        return total_cleared, combo_bonus

    def _create_particles(self, c, r, color):
        center_x, center_y = self._grid_to_pixel(c, r)
        for _ in range(10): # 10 particles per fruit
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = random.uniform(2, 5)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'radius': radius, 'color': color, 'life': 15})


    # --- Rendering Helpers ---

    def _grid_to_pixel(self, c, r):
        x = self.GRID_OFFSET_X + c * self.FRUIT_SIZE + self.FRUIT_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.FRUIT_SIZE + self.FRUIT_SIZE // 2
        return x, y

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, 
                         (self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_WIDTH, self.GRID_AREA_HEIGHT),
                         border_radius=8)

        # Draw fruits
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                fruit_type = self.grid[c, r]
                if fruit_type > 0:
                    px, py = self._grid_to_pixel(c, r)
                    color = self.FRUIT_COLORS[fruit_type - 1]
                    radius = self.FRUIT_SIZE // 2 - 5
                    
                    # Shadow
                    shadow_color = tuple(max(0, val - 40) for val in color)
                    pygame.gfxdraw.filled_circle(self.screen, int(px), int(py) + 2, radius, shadow_color)
                    
                    # Fruit
                    pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, color)
                    pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, color)

        # Draw selected fruit highlight
        if self.selected_pos:
            c, r = self.selected_pos
            px, py = self._grid_to_pixel(c, r)
            radius = self.FRUIT_SIZE // 2 - 2
            pygame.draw.circle(self.screen, self.COLOR_SELECT, (px, py), radius, 3)

        # Draw cursor
        c, r = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + c * self.FRUIT_SIZE,
                           self.GRID_OFFSET_Y + r * self.FRUIT_SIZE,
                           self.FRUIT_SIZE, self.FRUIT_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4, border_radius=6)

        # Draw and update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 15))))
                color = p['color'] + (alpha,)
                # Using a surface with per-pixel alpha for smooth fading
                particle_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(particle_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            board_cleared = np.all(self.grid == 0)
            msg = "PERFECT!" if board_cleared else "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with a no-op

    running = True
    while running:
        # --- Event Handling ---
        space_held = False
        shift_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
            
        action[1] = 1 if space_held else 0
        action[2] = 1 if shift_held else 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Reset action to no-op for next frame
        action[0] = 0

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for manual play

    env.close()