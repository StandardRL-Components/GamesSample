
# Generated: 2025-08-27T17:52:39.912557
# Source Brief: brief_01666.md
# Brief Index: 1666

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Create combos and chain reactions to maximize your score before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.GEM_TYPES = 5
        self.TARGET_SCORE = 1000
        self.MAX_MOVES = 50
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 80)
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECT = (255, 255, 255)
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 200, 50),   # Green
            (50, 100, 255),  # Blue
            (255, 200, 50),  # Yellow
            (180, 50, 255),  # Purple
        ]

        # Calculate board layout
        self.board_area_height = self.HEIGHT - 20
        self.gem_size = self.board_area_height // self.GRID_SIZE
        self.board_width = self.gem_size * self.GRID_SIZE
        self.board_offset_x = (self.WIDTH - self.board_width) // 2
        self.board_offset_y = (self.HEIGHT - self.gem_size * self.GRID_SIZE) // 2

        # Initialize state variables
        self.board = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.animations = None
        self.steps = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.last_action_time = 0

        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem_pos = None
        self.animations = []

        # Generate a board with at least one possible move
        while True:
            self._generate_board()
            if self._find_possible_moves():
                break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # If animations are playing, we don't process new actions, just advance animation time.
        # This makes the game feel smooth even though it's turn-based.
        if self._update_animations():
            terminated = self.game_over
            return self._get_observation(), 0, terminated, False, self._get_info()

        # If game is over, no more actions can be taken
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        
        # 2. Handle selection and swapping
        if space_pressed:
            # sfx: select_gem
            if self.selected_gem_pos is None:
                self.selected_gem_pos = tuple(self.cursor_pos)
            else:
                # Attempt to swap
                target_pos = tuple(self.cursor_pos)
                if self._is_adjacent(self.selected_gem_pos, target_pos):
                    self.moves_remaining -= 1
                    reward += self._perform_swap(self.selected_gem_pos, target_pos)
                else:
                    # sfx: invalid_selection
                    pass # Non-adjacent selection, just clear the first selection
                self.selected_gem_pos = None

        # 3. Check for termination conditions
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 100
            self.game_over = True
            terminated = True
            # sfx: victory
        elif self.moves_remaining <= 0:
            reward += -10
            self.game_over = True
            terminated = True
            # sfx: game_over_loss
        elif not self._find_possible_moves() and not self.animations:
            # No more moves left on the board
            reward += -10
            self.game_over = True
            terminated = True
            # sfx: game_over_stalemate
        
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

    def _perform_swap(self, pos1, pos2):
        # Swap gems in the data grid
        x1, y1 = pos1
        x2, y2 = pos2
        self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]

        # Check for matches
        matches = self._find_all_matches()
        if not matches:
            # Invalid swap, swap back
            self.board[y1, x1], self.board[y2, x2] = self.board[y2, x2], self.board[y1, x1]
            self._add_animation('swap_back', pos1=pos1, pos2=pos2, duration=200)
            # sfx: invalid_swap
            return -0.2
        else:
            # Valid swap, start cascade
            self._add_animation('swap', pos1=pos1, pos2=pos2, duration=200)
            # sfx: valid_swap
            reward = self._handle_cascade()
            return reward

    def _handle_cascade(self, combo_multiplier=1):
        total_reward = 0
        matches = self._find_all_matches()
        if not matches:
            return 0
        
        # sfx: match_found
        
        # Calculate score and reward for current matches
        num_matched = len(matches)
        reward_for_match = num_matched * combo_multiplier
        
        # Bonus for larger matches
        for match_set in self._group_matches(matches):
            if len(match_set) == 4:
                reward_for_match += 5
            elif len(match_set) >= 5:
                reward_for_match += 10
        
        self.score += int(reward_for_match * 10)
        total_reward += reward_for_match
        
        # Animate and remove matched gems
        for x, y in matches:
            self._add_animation('disappear', pos=(x, y), duration=200, delay=150, color=self.GEM_COLORS[self.board[y, x] - 1])
        
        # Apply gravity and refill, creating fall animations
        self._apply_gravity_and_refill(matches)
        
        # Recursively call for chain reactions (combos)
        total_reward += self._handle_cascade(combo_multiplier + 0.5)

        return total_reward

    def _apply_gravity_and_refill(self, matched_gems):
        cols_affected = sorted(list(set(x for x, y in matched_gems)))
        
        for x in cols_affected:
            col = self.board[:, x]
            empty_slots = 0
            new_col = np.zeros_like(col)
            new_col_idx = self.GRID_SIZE - 1
            
            # Move existing gems down
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if (x, y) not in matched_gems:
                    gem_type = self.board[y, x]
                    new_col[new_col_idx] = gem_type
                    if new_col_idx != y:
                        self._add_animation('fall', start_pos=(x, y), end_pos=(x, new_col_idx), duration=300, delay=350, gem_type=gem_type)
                    new_col_idx -= 1
                else:
                    empty_slots += 1
            
            # Refill with new gems
            for i in range(empty_slots):
                new_gem = self.np_random.integers(1, self.GEM_TYPES + 1)
                new_col[i] = new_gem
                self._add_animation('fall', start_pos=(x, -1 - i), end_pos=(x, i), duration=400, delay=350, gem_type=new_gem)
            
            self.board[:, x] = new_col

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
            "moves_remaining": self.moves_remaining,
            "game_over": self.game_over,
        }

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.board_offset_x + i * self.gem_size, self.board_offset_y)
            end_pos = (self.board_offset_x + i * self.gem_size, self.board_offset_y + self.GRID_SIZE * self.gem_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.board_offset_x, self.board_offset_y + i * self.gem_size)
            end_pos = (self.board_offset_x + self.GRID_SIZE * self.gem_size, self.board_offset_y + i * self.gem_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw gems
        rendered_gems = set()
        current_time = pygame.time.get_ticks()

        # Draw animated gems first
        for anim in self.animations:
            if anim['type'] in ['swap', 'swap_back', 'fall']:
                progress = min(1.0, (current_time - anim['start_time']) / anim['duration'])
                
                if anim['type'] == 'swap_back' and progress > 0.5:
                     progress = 1.0 - progress # Animate back to original spot
                
                if anim['type'] in ['swap', 'swap_back']:
                    pos1, pos2 = anim['pos1'], anim['pos2']
                    gem1_type = self.board[pos2[1], pos2[0]] if anim['type'] == 'swap' else self.board[pos1[1], pos1[0]]
                    gem2_type = self.board[pos1[1], pos1[0]] if anim['type'] == 'swap' else self.board[pos2[1], pos2[0]]
                    
                    self._draw_gem_at_pixel_pos(self._interp_pos(pos1, pos2, progress), gem1_type)
                    self._draw_gem_at_pixel_pos(self._interp_pos(pos2, pos1, progress), gem2_type)
                    rendered_gems.add(pos1)
                    rendered_gems.add(pos2)
                
                elif anim['type'] == 'fall':
                    self._draw_gem_at_pixel_pos(self._interp_pos(anim['start_pos'], anim['end_pos'], progress), anim['gem_type'])
                    rendered_gems.add(anim['end_pos'])

        # Draw static gems
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) not in rendered_gems and self.board[y, x] != 0:
                    self._draw_gem((x, y), self.board[y, x])

        # Draw particles from disappear animations
        for anim in self.animations:
            if anim['type'] == 'disappear':
                progress = min(1.0, (current_time - anim['start_time']) / anim['duration'])
                if progress < 1.0:
                    center_x, center_y = self._get_pixel_pos(anim['pos'])
                    for p in anim['particles']:
                        px = center_x + p['dir'][0] * progress * 25
                        py = center_y + p['dir'][1] * progress * 25
                        alpha = int(255 * (1 - progress))
                        color = anim['color']
                        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), p['size'], (*color, alpha))

    def _draw_gem(self, grid_pos, gem_type):
        pixel_pos = self._get_pixel_pos(grid_pos)
        self._draw_gem_at_pixel_pos(pixel_pos, gem_type)

    def _draw_gem_at_pixel_pos(self, pixel_pos, gem_type):
        if gem_type == 0: return
        center_x, center_y = int(pixel_pos[0]), int(pixel_pos[1])
        radius = int(self.gem_size * 0.4)
        color = self.GEM_COLORS[gem_type - 1]
        
        # Main gem body with antialiasing
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Highlight for 3D effect
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.filled_circle(self.screen, center_x - radius//3, center_y - radius//3, radius//4, highlight_color)

    def _render_ui(self):
        # Score and Moves
        score_surf = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        moves_surf = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))
        self.screen.blit(moves_surf, (self.WIDTH - moves_surf.get_width() - 15, 10))
        
        # Cursor
        if not self.game_over:
            cursor_rect = self._get_gem_rect(self.cursor_pos)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
        
        # Selection highlight
        if self.selected_gem_pos is not None:
            select_rect = self._get_gem_rect(self.selected_gem_pos)
            
            # Pulsing effect
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
            width = int(2 + pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, select_rect.inflate(4, 4), width, border_radius=6)
        
        # Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    # --- Helper Functions ---

    def _generate_board(self):
        self.board = self.np_random.integers(1, self.GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        # Ensure no initial matches
        while self._find_all_matches():
            matches = self._find_all_matches()
            for x, y in matches:
                self.board[y, x] = self.np_random.integers(1, self.GEM_TYPES + 1)

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Horizontal check
                if x < self.GRID_SIZE - 2 and self.board[y, x] == self.board[y, x+1] == self.board[y, x+2] != 0:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical check
                if y < self.GRID_SIZE - 2 and self.board[y, x] == self.board[y+1, x] == self.board[y+2, x] != 0:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches
    
    def _group_matches(self, match_coords):
        # Helper to group connected matched gems for scoring (e.g., L-shapes)
        if not match_coords:
            return []
        
        groups = []
        coords = set(match_coords)
        while coords:
            current_group = set()
            q = [coords.pop()]
            current_group.add(q[0])
            
            head = 0
            while head < len(q):
                x, y = q[head]
                head += 1
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    neighbor = (x + dx, y + dy)
                    if neighbor in coords:
                        coords.remove(neighbor)
                        current_group.add(neighbor)
                        q.append(neighbor)
            groups.append(current_group)
        return groups

    def _find_possible_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Try swapping right
                if x < self.GRID_SIZE - 1:
                    self.board[y, x], self.board[y, x+1] = self.board[y, x+1], self.board[y, x]
                    if self._find_all_matches():
                        self.board[y, x], self.board[y, x+1] = self.board[y, x+1], self.board[y, x]
                        return True
                    self.board[y, x], self.board[y, x+1] = self.board[y, x+1], self.board[y, x]
                # Try swapping down
                if y < self.GRID_SIZE - 1:
                    self.board[y, x], self.board[y+1, x] = self.board[y+1, x], self.board[y, x]
                    if self._find_all_matches():
                        self.board[y, x], self.board[y+1, x] = self.board[y+1, x], self.board[y, x]
                        return True
                    self.board[y, x], self.board[y+1, x] = self.board[y+1, x], self.board[y, x]
        return False

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _get_pixel_pos(self, grid_pos):
        x, y = grid_pos
        px = self.board_offset_x + x * self.gem_size + self.gem_size / 2
        py = self.board_offset_y + y * self.gem_size + self.gem_size / 2
        return px, py

    def _get_gem_rect(self, grid_pos):
        x, y = grid_pos
        return pygame.Rect(
            self.board_offset_x + x * self.gem_size,
            self.board_offset_y + y * self.gem_size,
            self.gem_size, self.gem_size
        )
    
    def _interp_pos(self, pos1, pos2, progress):
        p1 = self._get_pixel_pos(pos1)
        p2 = self._get_pixel_pos(pos2)
        return p1[0] + (p2[0] - p1[0]) * progress, p1[1] + (p2[1] - p1[1]) * progress

    def _add_animation(self, anim_type, **kwargs):
        anim = {
            'type': anim_type,
            'start_time': pygame.time.get_ticks() + kwargs.get('delay', 0),
            'duration': kwargs['duration'],
        }
        if anim_type == 'disappear':
            anim['pos'] = kwargs['pos']
            anim['color'] = kwargs['color']
            anim['particles'] = [{'dir': (random.uniform(-1, 1), random.uniform(-1, 1)), 'size': random.randint(2, 4)} for _ in range(10)]
        else:
            anim.update(kwargs)
        self.animations.append(anim)

    def _update_animations(self):
        current_time = pygame.time.get_ticks()
        active_animations = []
        finished_something = False
        for anim in self.animations:
            if current_time >= anim['start_time']:
                if current_time < anim['start_time'] + anim['duration']:
                    active_animations.append(anim)
                else:
                    finished_something = True
            else:
                active_animations.append(anim)
        
        if finished_something and not active_animations:
            # Last animation just finished. If no moves are possible, end game.
            if not self.game_over and not self._find_possible_moves():
                self.game_over = True
                # sfx: game_over_stalemate
        
        self.animations = active_animations
        return bool(self.animations)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        action = [movement, space_pressed, 0] # Movement, Space, Shift
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If there are animations, we need to keep stepping to see them
        while env.animations:
            obs, _, _, _, _ = env.step([0, 0, 0]) # Step with no-op to advance animation
            # Render to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(60) # Run animations at 60fps

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Keep displaying the final screen until a new game is started or quit
        
        # Render to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate for human play

    pygame.quit()