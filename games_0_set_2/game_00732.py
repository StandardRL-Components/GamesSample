
# Generated: 2025-08-27T14:36:34.747643
# Source Brief: brief_00732.md
# Brief Index: 732

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle swap direction (arrow indicator). Press Space to perform the swap."
    )

    game_description = (
        "A strategic match-3 puzzle game. Swap gems to create matches of three or more, and collect 10 gems before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    GEM_TYPES = 4
    GEM_GOAL = 10
    MAX_MOVES = 20
    MAX_STEPS = 1000 # Safety break

    # Rewards
    REWARD_MATCH_PER_GEM = 1
    REWARD_BONUS_MATCH = 5
    REWARD_INVALID_SWAP = -0.1
    REWARD_WIN = 50
    REWARD_LOSS = -50

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_BG = (30, 35, 55)
    COLOR_GRID_LINE = (50, 55, 75)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 220, 80),  # Yellow
    ]

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.grid_pixel_width = self.SCREEN_HEIGHT - 40
        self.cell_size = self.grid_pixel_width // self.GRID_WIDTH
        self.grid_pixel_width = self.cell_size * self.GRID_WIDTH # Recalculate to avoid gaps
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_width) // 2

        self.grid = None
        self.cursor_pos = None
        self.swap_direction = None
        self.moves_left = None
        self.gems_collected = None
        self.game_over = None
        self.steps = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = None

        self.reset()
        # self.validate_implementation() # For testing during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.gems_collected = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.swap_direction = 0  # 0: up, 1: right, 2: down, 3: left
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        self._generate_initial_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        
        movement, space_press, shift_press = self._process_actions(action)

        if movement != 0: self._move_cursor(movement)
        if shift_press: self._cycle_swap_direction()
        if space_press:
            reward, terminated = self._execute_swap()

        self._update_particles()
        
        # Check for loss condition if not already terminated by a win
        if not terminated and self.moves_left <= 0 and self.gems_collected < self.GEM_GOAL:
            terminated = True
            reward = self.REWARD_LOSS
            self.game_over = True
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _process_actions(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return movement, space_press, shift_press

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        # Wrap around
        self.cursor_pos[0] %= self.GRID_WIDTH
        self.cursor_pos[1] %= self.GRID_HEIGHT
        
    def _cycle_swap_direction(self):
        self.swap_direction = (self.swap_direction + 1) % 4
        # SFX: UI_Bleep

    def _execute_swap(self):
        if self.moves_left <= 0:
            return 0, self.game_over

        self.moves_left -= 1
        # SFX: Swap_Sound
        
        cx, cy = self.cursor_pos
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.swap_direction]
        nx, ny = cx + dx, cy + dy

        if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
            return self.REWARD_INVALID_SWAP, self.game_over

        # Perform swap
        self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]

        total_reward = 0
        total_gems_cleared = 0
        
        # Cascade loop
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            num_gems_in_cascade = len(matches)
            total_gems_cleared += num_gems_in_cascade
            
            cascade_reward = num_gems_in_cascade * self.REWARD_MATCH_PER_GEM
            if num_gems_in_cascade >= 4:
                cascade_reward += self.REWARD_BONUS_MATCH
            total_reward += cascade_reward

            # SFX: Match_Success
            for r, c in matches:
                self._create_particles(c, r, self.grid[r, c])
            
            self._clear_gems(matches)
            self._apply_gravity()
            self._refill_grid()

        if total_gems_cleared == 0:
            # No match found, swap back and penalize
            self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
            return self.REWARD_INVALID_SWAP, self.game_over
        else:
            self.gems_collected += total_gems_cleared
            if self.gems_collected >= self.GEM_GOAL:
                self.game_over = True
                # SFX: Game_Win
                return total_reward + self.REWARD_WIN, True

        return total_reward, self.game_over

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == -1: continue

                # Horizontal
                h_match = [c]
                for i in range(c + 1, self.GRID_WIDTH):
                    if self.grid[r, i] == gem_type: h_match.append(i)
                    else: break
                if len(h_match) >= 3:
                    for i in h_match: matches.add((r, i))

                # Vertical
                v_match = [r]
                for i in range(r + 1, self.GRID_HEIGHT):
                    if self.grid[i, c] == gem_type: v_match.append(i)
                    else: break
                if len(v_match) >= 3:
                    for i in v_match: matches.add((i, c))
        return matches

    def _clear_gems(self, matches):
        for r, c in matches:
            self.grid[r, c] = -1 # Mark as empty

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _refill_grid(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.GEM_TYPES)

    def _generate_initial_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            while self._find_all_matches():
                matches = self._find_all_matches()
                self._clear_gems(matches)
                self._apply_gravity()
                self._refill_grid()
            
            if self._has_possible_move():
                break

    def _has_possible_move(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Test swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_all_matches():
                        self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                # Test swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_all_matches():
                        self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.gems_collected,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "game_over": self.game_over,
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_pixel_width, self.grid_pixel_width)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    self._render_gem(c, r, gem_type)
        
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'] + (p['alpha'],))

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cx * self.cell_size,
            self.grid_offset_y + cy * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=8)

        # Draw swap indicator
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.swap_direction]
        arrow_start_x = self.grid_offset_x + (cx + 0.5) * self.cell_size
        arrow_start_y = self.grid_offset_y + (cy + 0.5) * self.cell_size
        arrow_end_x = arrow_start_x + dx * self.cell_size * 0.4
        arrow_end_y = arrow_start_y + dy * self.cell_size * 0.4
        self._draw_arrow(self.screen, self.COLOR_CURSOR, (arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y))
    
    def _render_gem(self, c, r, gem_type):
        center_x = self.grid_offset_x + int((c + 0.5) * self.cell_size)
        center_y = self.grid_offset_y + int((r + 0.5) * self.cell_size)
        radius = int(self.cell_size * 0.4)
        base_color = self.GEM_COLORS[gem_type]
        
        # Shadow
        shadow_color = (max(0, base_color[0]-80), max(0, base_color[1]-80), max(0, base_color[2]-80))
        pygame.gfxdraw.filled_circle(self.screen, center_x + 2, center_y + 2, radius, shadow_color)

        # Base color
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, base_color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, base_color)
        
        # Highlight
        highlight_color = (min(255, base_color[0]+100), min(255, base_color[1]+100), min(255, base_color[2]+100))
        pygame.gfxdraw.filled_circle(self.screen, center_x - radius//3, center_y - radius//3, radius//3, highlight_color + (100,))

    def _render_ui(self):
        # Moves Left
        moves_text = f"Moves: {self.moves_left}"
        self._draw_text(moves_text, self.font_small, (20, 20))
        
        # Gems Collected
        gems_text = f"Collected: {self.gems_collected} / {self.GEM_GOAL}"
        self._draw_text(gems_text, self.font_small, (20, 50))
        
        # Game Over Message
        if self.game_over:
            if self.gems_collected >= self.GEM_GOAL:
                msg = "YOU WIN!"
                color = self.GEM_COLORS[3]
            else:
                msg = "GAME OVER"
                color = self.GEM_COLORS[0]
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            self._draw_text(msg, self.font_large, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), color=color, center=True)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        text_shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(text_shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_arrow(self, screen, color, start, end, width=3):
        pygame.draw.line(screen, color, start, end, width)
        angle = math.atan2(start[1] - end[1], start[0] - end[0])
        p1 = (end[0] + 8 * math.cos(angle - math.pi / 6), end[1] + 8 * math.sin(angle - math.pi / 6))
        p2 = (end[0] + 8 * math.cos(angle + math.pi / 6), end[1] + 8 * math.sin(angle + math.pi / 6))
        pygame.draw.polygon(screen, color, [end, p1, p2])

    def _create_particles(self, c, r, gem_type):
        center_x = self.grid_offset_x + (c + 0.5) * self.cell_size
        center_y = self.grid_offset_y + (r + 0.5) * self.cell_size
        base_color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'color': base_color,
                'alpha': 255,
                'life': random.uniform(10, 20)
            })

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['alpha'] = max(0, int(255 * (p['life'] / 20)))
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    done = False
        
        action = [movement, space, shift]
        
        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
            if terminated:
                print("Game Over!")
                print(f"Final Score: {info['score']}")
                done = True

        # --- Rendering ---
        # The observation is already a rendered frame
        # Convert it back to a Pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()