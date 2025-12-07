
# Generated: 2025-08-28T01:01:23.824741
# Source Brief: brief_03974.md
# Brief Index: 3974

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap. Press Shift to cancel a selection."
    )

    game_description = (
        "An isometric match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Clear the entire board before you run out of moves to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BOARD_SIZE = 8
    MAX_MOVES = 15
    INITIAL_COLORS = 5

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)

    GEM_COLORS = [
        (255, 50, 50),    # Red
        (50, 255, 50),    # Green
        (50, 150, 255),   # Blue
        (255, 150, 50),   # Orange
        (200, 50, 255),   # Purple
        (50, 255, 255),   # Cyan
        (255, 255, 100),  # Yellow
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.tile_width = 48
        self.tile_height = self.tile_width // 2
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 100

        self.board = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_left = 0
        self.score = 0
        self.stage = 1
        self.num_colors = 0
        self.game_over = False
        self.win_state = False

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.animations = []
        self.particles = []

        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.stage = options.get("stage", 1) if options else 1
        self.num_colors = min(len(self.GEM_COLORS), self.INITIAL_COLORS - 1 + self.stage)
        
        self._generate_board()

        self.cursor_pos = [self.BOARD_SIZE // 2, self.BOARD_SIZE // 2]
        self.selected_gem = None
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win_state = False

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.animations = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        move_made = False

        # --- Unpack Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Handle Input ---
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        if movement == 2: self.cursor_pos[0] = min(self.BOARD_SIZE - 1, self.cursor_pos[0] + 1)
        if movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        if movement == 4: self.cursor_pos[1] = min(self.BOARD_SIZE - 1, self.cursor_pos[1] + 1)

        if shift_press and self.selected_gem:
            self.selected_gem = None
            # sfx: cancel_select

        if space_press:
            cursor_r, cursor_c = self.cursor_pos
            if self.board[cursor_r, cursor_c] == -1: # Cannot select empty space
                self.selected_gem = None
                # sfx: error_sound
            elif self.selected_gem is None:
                self.selected_gem = [cursor_r, cursor_c]
                # sfx: select_gem
            else:
                sel_r, sel_c = self.selected_gem
                dist = abs(sel_r - cursor_r) + abs(sel_c - cursor_c)
                if dist == 1: # Is adjacent
                    move_made = True
                    self.moves_left -= 1
                    
                    # Perform swap
                    self.board[sel_r, sel_c], self.board[cursor_r, cursor_c] = \
                        self.board[cursor_r, cursor_c], self.board[sel_r, sel_c]
                    
                    self._add_animation('swap', (sel_r, sel_c), (cursor_r, cursor_c), duration=15)
                    self._run_animations()
                    
                    total_cleared_gems = self._process_matches()
                    
                    if total_cleared_gems > 0:
                        reward += total_cleared_gems
                        self.score += total_cleared_gems
                        # sfx: match_success
                    else: # No match, swap back
                        reward -= 0.1
                        self.board[sel_r, sel_c], self.board[cursor_r, cursor_c] = \
                            self.board[cursor_r, cursor_c], self.board[sel_r, sel_c]
                        self._add_animation('swap', (sel_r, sel_c), (cursor_r, cursor_c), duration=15)
                        self._run_animations()
                        # sfx: invalid_swap

                    self.selected_gem = None
                else: # Not adjacent
                    self.selected_gem = [cursor_r, cursor_c] # Select the new gem instead
                    # sfx: select_gem
        
        if move_made and not self._has_possible_moves() and np.any(self.board != -1):
            self._reshuffle_board()
            self._add_animation('shuffle', duration=30)
            self._run_animations()

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win_state:
                reward += 100
                self.score += 100
            else:
                reward = -100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_matches(self):
        total_cleared_this_turn = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            num_cleared = len(matches)
            total_cleared_this_turn += num_cleared
            
            # Clear gems and create particles
            for r, c in matches:
                if self.board[r, c] != -1:
                    self._create_particles(r, c, self.board[r,c])
                    self.board[r, c] = -1
            
            self._add_animation('destroy', matches, duration=20)
            self._run_animations()
            
            # Apply gravity
            self._apply_gravity()
            self._add_animation('fall', duration=20)
            self._run_animations()

        return total_cleared_this_turn

    def _check_termination(self):
        if np.all(self.board == -1):
            self.win_state = True
            return True
        if self.moves_left <= 0:
            self.win_state = False
            return True
        return False

    def _generate_board(self):
        self.board = self.np_random.integers(0, self.num_colors, size=(self.BOARD_SIZE, self.BOARD_SIZE))
        while not self._has_possible_moves() or len(self._find_matches()) > 0:
            self.board = self.np_random.integers(0, self.num_colors, size=(self.BOARD_SIZE, self.BOARD_SIZE))

    def _reshuffle_board(self):
        non_empty_indices = np.argwhere(self.board != -1)
        non_empty_values = self.board[self.board != -1]
        
        self.np_random.shuffle(non_empty_values)
        
        new_board = np.full_like(self.board, -1)
        for idx, val in zip(non_empty_indices, non_empty_values):
            new_board[tuple(idx)] = val
        
        self.board = new_board
        
        if not self._has_possible_moves() and np.any(self.board != -1):
            self._generate_board() # Failsafe, should be rare

    def _find_matches(self):
        matches = set()
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == -1:
                    continue
                # Horizontal
                if c < self.BOARD_SIZE - 2 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.BOARD_SIZE - 2 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _has_possible_moves(self):
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == -1:
                    continue
                # Check swap right
                if c < self.BOARD_SIZE - 1:
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                    if len(self._find_matches()) > 0:
                        self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                        return True
                    self.board[r, c], self.board[r, c+1] = self.board[r, c+1], self.board[r, c]
                # Check swap down
                if r < self.BOARD_SIZE - 1:
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
                    if len(self._find_matches()) > 0:
                        self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
                        return True
                    self.board[r, c], self.board[r+1, c] = self.board[r+1, c], self.board[r, c]
        return False

    def _apply_gravity(self):
        for c in range(self.BOARD_SIZE):
            empty_row = self.BOARD_SIZE - 1
            for r in range(self.BOARD_SIZE - 1, -1, -1):
                if self.board[r, c] != -1:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = -1
                    empty_row -= 1
    
    def _iso_to_screen(self, r, c):
        x = self.origin_x + (c - r) * self.tile_width / 2
        y = self.origin_y + (c + r) * self.tile_height / 2
        return int(x), int(y)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "stage": self.stage,
        }

    def _render_game(self):
        # Draw grid
        for r in range(self.BOARD_SIZE + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.BOARD_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.BOARD_SIZE + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.BOARD_SIZE, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Draw gems
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                gem_type = self.board[r, c]
                if gem_type != -1:
                    self._draw_gem(r, c, self.GEM_COLORS[gem_type])
        
        # Draw selections
        if self.selected_gem:
            r, c = self.selected_gem
            self._draw_highlight(r, c, self.COLOR_SELECTED, 3)

        # Draw cursor
        self._draw_highlight(self.cursor_pos[0], self.cursor_pos[1], self.COLOR_CURSOR, 2)
        
        # Draw particles
        self._update_and_draw_particles()

    def _draw_gem(self, r, c, color, scale=1.0):
        x, y = self._iso_to_screen(r, c)
        w, h = self.tile_width * scale, self.tile_height * scale
        
        points = [
            (x, y - h / 2),
            (x + w / 2, y),
            (x, y + h / 2),
            (x - w / 2, y)
        ]
        
        # Use integer points for pygame drawing
        int_points = [(int(px), int(py)) for px, py in points]
        
        light_color = tuple(min(255, val + 60) for val in color)
        dark_color = tuple(max(0, val - 60) for val in color)

        pygame.gfxdraw.filled_polygon(self.screen, int_points, color)
        pygame.gfxdraw.aapolygon(self.screen, int_points, light_color)

        # 3D effect
        top_face = [int_points[0], int_points[1], (int_points[1][0], int_points[1][1] - 3), (int_points[0][0], int_points[0][1] - 3)]
        pygame.gfxdraw.filled_polygon(self.screen, top_face, light_color)
        pygame.gfxdraw.aapolygon(self.screen, top_face, light_color)

    def _draw_highlight(self, r, c, color, width):
        x, y = self._iso_to_screen(r, c)
        w, h = self.tile_width, self.tile_height
        points = [
            (x, y - h / 2), (x + w / 2, y),
            (x, y + h / 2), (x - w / 2, y)
        ]
        int_points = [(int(px), int(py)) for px, py in points]
        pygame.draw.lines(self.screen, color, True, int_points, width)

    def _render_ui(self):
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))

        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        stage_text = self.font_small.render(f"Stage: {self.stage}", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(midtop=(self.SCREEN_WIDTH // 2, 10))
        self.screen.blit(stage_text, stage_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win_state else "GAME OVER"
            color = (100, 255, 100) if self.win_state else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _add_animation(self, anim_type, *args, **kwargs):
        self.animations.append({'type': anim_type, 'args': args, 'kwargs': kwargs, 'progress': 0})
        
    def _run_animations(self):
        while self.animations:
            anim = self.animations[0]
            anim['progress'] += 1
            
            # This is a dummy loop for visualization, as rgb_array only needs the final frame.
            # In a human-render mode, we would draw here.
            # We'll just fast-forward the animation.
            if anim['progress'] >= anim['kwargs'].get('duration', 1):
                self.animations.pop(0)

    def _create_particles(self, r, c, gem_type):
        x, y = self._iso_to_screen(r, c)
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(20, 40)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = 255 * (p['life'] / p['max_life'])
                color = (*p['color'], alpha)
                size = 2 * (p['life'] / p['max_life'])
                rect = pygame.Rect(p['pos'][0] - size, p['pos'][1] - size, size*2, size*2)
                
                # Create a temporary surface for the particle to handle alpha
                particle_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, rect.topleft)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gemstone Glade")
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    while running:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Action is applied only when a key is pressed to suit auto_advance=False
        if mov != 0 or space != env.prev_space_held or shift != env.prev_shift_held:
            action = [mov, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    env.close()