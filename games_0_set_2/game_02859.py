
# Generated: 2025-08-28T06:11:24.661505
# Source Brief: brief_02859.md
# Brief Index: 2859

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a gem. "
        "Move to an adjacent gem and press shift to swap."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Clear the entire board before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 15
    GRID_ROWS = 5
    GEM_COUNT = GRID_COLS * GRID_ROWS  # 75 gems
    NUM_GEM_TYPES = 6
    MAX_MOVES = 30
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 160, 80),   # Orange
    ]
    
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_SELECTION = (255, 255, 255)

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
        
        self.grid_pixel_width = self.SCREEN_WIDTH - 40
        self.grid_pixel_height = self.SCREEN_HEIGHT - 200
        self.gem_size = min(self.grid_pixel_width // self.GRID_COLS, self.grid_pixel_height // self.GRID_ROWS)
        self.grid_origin_x = (self.SCREEN_WIDTH - self.GRID_COLS * self.gem_size) // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.GRID_ROWS * self.gem_size) // 2 + 50
        
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_gem_pos = None
        self.particles = []
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._handle_cursor_movement(movement)
        
        if space_held:
            # SFX: select_gem.wav
            self.selected_gem_pos = tuple(self.cursor_pos)
            
        if shift_held and self.selected_gem_pos:
            reward = self._attempt_swap()

        self.steps += 1
        
        # Check for termination conditions
        board_cleared = np.sum(self.grid) == 0
        if self.moves_left <= 0 or board_cleared or self.steps >= self.MAX_STEPS:
            self.game_over = True
            if board_cleared:
                reward += 100 # Win bonus
                # SFX: win_game.wav
            else:
                reward += -50 # Loss penalty
                # SFX: lose_game.wav

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _handle_cursor_movement(self, movement):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

    def _attempt_swap(self):
        target_pos = tuple(self.cursor_pos)
        
        # Check for adjacency
        dx = abs(self.selected_gem_pos[0] - target_pos[0])
        dy = abs(self.selected_gem_pos[1] - target_pos[1])
        
        if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
            self.moves_left -= 1
            # SFX: swap.wav
            self._swap_gems(self.selected_gem_pos, target_pos)
            
            total_matches, reward = self._find_and_process_matches()
            
            if total_matches == 0:
                # No match, penalize and swap back
                self._swap_gems(self.selected_gem_pos, target_pos) # Undo swap
                # SFX: invalid_swap.wav
                self.selected_gem_pos = None
                return -0.2
            
            self.selected_gem_pos = None
            return reward
        else:
            # Not adjacent, reset selection
            self.selected_gem_pos = None
            return 0

    def _swap_gems(self, pos1, pos2):
        self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]

    def _find_and_process_matches(self):
        total_reward = 0
        total_gems_matched = 0
        
        while True:
            matches = self._find_matches()
            if not matches:
                break

            num_gems_in_chain = len(matches)
            total_gems_matched += num_gems_in_chain
            total_reward += num_gems_in_chain # +1 per gem
            
            # Bonus for larger matches
            match_lengths = self._get_match_lengths(matches)
            if any(length >= 5 for length in match_lengths):
                total_reward += 10
                # SFX: match_5.wav
            elif any(length >= 4 for length in match_lengths):
                total_reward += 5
                # SFX: match_4.wav
            else:
                # SFX: match_3.wav
                pass
            
            self.score += int(total_reward)

            for x, y in matches:
                self._create_particles(x, y, self.grid[x, y])
                self.grid[x, y] = 0 # 0 represents an empty space
            
            self._apply_gravity()
            self._fill_top_rows()
            
        return total_gems_matched, total_reward
        
    def _find_matches(self):
        matches = set()
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                if self.grid[x, y] == 0: continue
                # Horizontal
                if x < self.GRID_COLS - 2 and self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical
                if y < self.GRID_ROWS - 2 and self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches
        
    def _get_match_lengths(self, matches):
        # A simplified way to estimate match lengths for bonuses
        lengths = []
        visited = set()
        for x, y in matches:
            if (x, y) in visited: continue
            
            # Horizontal check
            h_len = 1
            for i in range(1, self.GRID_COLS):
                if (x+i, y) in matches and (x+i, y) not in visited: h_len += 1; visited.add((x+i, y))
                else: break
            for i in range(1, self.GRID_COLS):
                if (x-i, y) in matches and (x-i, y) not in visited: h_len += 1; visited.add((x-i, y))
                else: break
            if h_len >= 3: lengths.append(h_len)
            
            # Vertical check
            v_len = 1
            for i in range(1, self.GRID_ROWS):
                if (x, y+i) in matches and (x, y+i) not in visited: v_len += 1; visited.add((x, y+i))
                else: break
            for i in range(1, self.GRID_ROWS):
                if (x, y-i) in matches and (x, y-i) not in visited: v_len += 1; visited.add((x, y-i))
                else: break
            if v_len >= 3: lengths.append(v_len)
            
            visited.add((x,y))
        return lengths

    def _apply_gravity(self):
        for x in range(self.GRID_COLS):
            empty_slots = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = 0

    def _fill_top_rows(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_COLS, self.GRID_ROWS))
            while self._find_matches():
                self._find_and_process_matches()
            if self._check_for_possible_moves():
                break
    
    def _check_for_possible_moves(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                # Swap right
                if x < self.GRID_COLS - 1:
                    self._swap_gems((x, y), (x + 1, y))
                    if self._find_matches():
                        self._swap_gems((x, y), (x + 1, y))
                        return True
                    self._swap_gems((x, y), (x + 1, y))
                # Swap down
                if y < self.GRID_ROWS - 1:
                    self._swap_gems((x, y), (x, y + 1))
                    if self._find_matches():
                        self._swap_gems((x, y), (x, y + 1))
                        return True
                    self._swap_gems((x, y), (x, y + 1))
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_origin_x, self.grid_origin_y, 
                                self.GRID_COLS * self.gem_size, self.GRID_ROWS * self.gem_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=8)

        # Update and draw particles
        self._update_and_draw_particles()

        # Draw gems
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                gem_type = self.grid[x, y]
                if gem_type > 0:
                    self._draw_gem(x, y, gem_type)

        # Draw selection highlight
        if self.selected_gem_pos:
            x, y = self.selected_gem_pos
            center_x = self.grid_origin_x + x * self.gem_size + self.gem_size // 2
            center_y = self.grid_origin_y + y * self.gem_size + self.gem_size // 2
            radius = self.gem_size // 2
            
            # Pulsating effect
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius * (1 + pulse * 0.1)), self.COLOR_SELECTION)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius * (1 + pulse * 0.1))+1, self.COLOR_SELECTION)

        # Draw cursor
        cursor_rect = pygame.Rect(self.grid_origin_x + self.cursor_pos[0] * self.gem_size,
                                  self.grid_origin_y + self.cursor_pos[1] * self.gem_size,
                                  self.gem_size, self.gem_size)
        
        surf = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, self.COLOR_CURSOR, surf.get_rect(), 5, border_radius=6)
        self.screen.blit(surf, cursor_rect.topleft)

    def _draw_gem(self, x, y, gem_type):
        center_x = self.grid_origin_x + x * self.gem_size + self.gem_size // 2
        center_y = self.grid_origin_y + y * self.gem_size + self.gem_size // 2
        radius = self.gem_size // 2 - 4
        
        color = self.GEM_COLORS[gem_type - 1]
        highlight = tuple(min(255, c + 60) for c in color)
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        
        # Add a shine effect
        shine_x = center_x - radius // 3
        shine_y = center_y - radius // 3
        pygame.gfxdraw.filled_circle(self.screen, shine_x, shine_y, radius // 3, highlight)
        pygame.gfxdraw.aacircle(self.screen, shine_x, shine_y, radius // 3, highlight)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_large, self.COLOR_TEXT, (20, 20))

        # Moves
        moves_text = f"MOVES: {self.moves_left}"
        text_width = self.font_large.size(moves_text)[0]
        draw_text(moves_text, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 20, 20))
        
        if self.game_over:
            board_cleared = np.sum(self.grid) == 0
            message = "BOARD CLEARED!" if board_cleared else "OUT OF MOVES"
            
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            text_width, text_height = self.font_large.size(message)
            draw_text(message, self.font_large, self.COLOR_TEXT, 
                      ((self.SCREEN_WIDTH - text_width) / 2, (self.SCREEN_HEIGHT - text_height) / 2 - 30))
            
            final_score_text = f"FINAL SCORE: {self.score}"
            text_width, text_height = self.font_small.size(final_score_text)
            draw_text(final_score_text, self.font_small, self.COLOR_TEXT, 
                      ((self.SCREEN_WIDTH - text_width) / 2, (self.SCREEN_HEIGHT - text_height) / 2 + 30))

    def _create_particles(self, grid_x, grid_y, gem_type):
        center_x = self.grid_origin_x + grid_x * self.gem_size + self.gem_size // 2
        center_y = self.grid_origin_y + grid_y * self.gem_size + self.gem_size // 2
        color = self.GEM_COLORS[gem_type - 1]
        
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 40)
            size = random.randint(3, 6)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'color': color, 'size': size})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            
            if p['life'] > 0:
                alpha = int(255 * (p['life'] / 40))
                color = (*p['color'], alpha)
                
                surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(surf, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))
                
                active_particles.append(p)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "gems_on_board": self.GEM_COUNT - np.count_nonzero(self.grid==0),
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)

    while not done:
        # --- Action mapping for human play ---
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()