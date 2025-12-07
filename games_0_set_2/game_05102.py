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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, then move to an adjacent "
        "gem and press Space again to swap. Match 3 or more gems to score."
    )

    game_description = (
        "A colorful gem-matching puzzle game. Strategically swap gems to create matches of three or more. "
        "Create chain reactions to maximize your score. Clear 50 gems before you run out of moves to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_WIDTH, self.BOARD_HEIGHT = 10, 8
        self.GEM_TYPES = 6
        self.GRID_START_X = (self.WIDTH - self.BOARD_WIDTH * 40) // 2
        self.GRID_START_Y = (self.HEIGHT - self.BOARD_HEIGHT * 40) // 2
        self.CELL_SIZE = 40
        self.GEM_SIZE = 16
        self.WIN_GEM_TARGET = 50
        self.MOVE_LIMIT = 20
        self.MAX_STEPS = 30 * 60  # 60 seconds at 30fps

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_SCORE = (255, 215, 0)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (100, 200, 255)
        self.GEM_COLORS = [
            (255, 80, 80),  # Red
            (80, 255, 80),  # Green
            (80, 150, 255),  # Blue
            (255, 150, 50),  # Orange
            (200, 80, 255),  # Purple
            (255, 255, 100),  # Yellow
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_game_over = pygame.font.SysFont("Verdana", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 60)

        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_remaining = None
        self.gems_cleared = None
        self.score = None
        self.game_over = None
        self.win_status = None
        self.game_state = None
        self.animations = None
        self.particles = None
        self.last_space_state = 0
        self.steps = 0

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._populate_board()
        self.cursor_pos = [0, 0]
        self.selected_gem = None
        self.moves_remaining = self.MOVE_LIMIT
        self.gems_cleared = 0
        self.score = 0
        self.game_over = False
        self.win_status = None  # None, 'win', or 'loss'
        self.game_state = 'INPUT'  # 'INPUT', 'ANIMATING'
        self.animations = []
        self.particles = []
        self.last_space_state = 0
        self.steps = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_state
        self.last_space_state = space_held

        if not self.game_over:
            if self.game_state == 'INPUT':
                reward += self._handle_input(movement, space_pressed)

            if self.game_state == 'ANIMATING':
                reward += self._update_animations()

            # Check for termination conditions
            if self.gems_cleared >= self.WIN_GEM_TARGET:
                self.game_over = True
                self.win_status = 'win'
                reward += 100
            elif self.moves_remaining <= 0 and self.game_state == 'INPUT':
                self.game_over = True
                self.win_status = 'loss'
                reward -= 100
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                self.win_status = 'loss'  # Time out
                reward -= 100

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        if movement == 2: self.cursor_pos[1] = min(self.BOARD_HEIGHT - 1, self.cursor_pos[1] + 1)
        if movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        if movement == 4: self.cursor_pos[0] = min(self.BOARD_WIDTH - 1, self.cursor_pos[0] + 1)

        # Handle selection/swap
        if space_pressed:
            cx, cy = self.cursor_pos
            if self.selected_gem is None:
                self.selected_gem = (cx, cy)
                # Sound: select_gem.wav
            else:
                sx, sy = self.selected_gem
                if abs(sx - cx) + abs(sy - cy) == 1:  # Is adjacent
                    self.moves_remaining -= 1
                    self._create_swap_animation((sx, sy), (cx, cy))
                    self.game_state = 'ANIMATING'
                    self.selected_gem = None
                else:  # Not adjacent, just change selection
                    self.selected_gem = (cx, cy)
                    # Sound: select_gem.wav
        return 0

    def _update_animations(self):
        reward = 0
        if not self.animations:
            self._start_match_check()
            if not self.animations:  # If no matches were found after checks
                self.game_state = 'INPUT'
            return reward

        finished_animations = []
        for anim in self.animations:
            anim['progress'] += 1
            if anim['progress'] >= anim['duration']:
                finished_animations.append(anim)

        for anim in finished_animations:
            self.animations.remove(anim)
            if anim['type'] == 'swap':
                p1, p2 = anim['p1'], anim['p2']
                self.grid[p1[1]][p1[0]], self.grid[p2[1]][p2[0]] = self.grid[p2[1]][p2[0]], self.grid[p1[1]][p1[0]]
                # After swap, check for matches
                matches = self._find_matches()
                if not matches:  # Invalid move, swap back
                    self._create_swap_animation(p1, p2, is_rewind=True)
                    self.moves_remaining += 1  # Refund move
                    reward -= 0.2
                    # Sound: invalid_swap.wav
                else:  # Valid move, start clearing
                    reward += self._handle_matches(matches)
                    # Sound: match_success.wav

            elif anim['type'] == 'rewind_swap':
                p1, p2 = anim['p1'], anim['p2']
                self.grid[p1[1]][p1[0]], self.grid[p2[1]][p2[0]] = self.grid[p2[1]][p2[0]], self.grid[p1[1]][p1[0]]

            elif anim['type'] == 'clear' and not self.animations:
                self._apply_gravity_and_refill()

            elif anim['type'] == 'fall' and not self.animations:
                self._start_match_check()

        if not self.animations:
            self.game_state = 'INPUT'

        return reward

    def _start_match_check(self):
        matches = self._find_matches()
        if matches:
            self._handle_matches(matches)
            # Sound: cascade_match.wav

    def _find_matches(self):
        matches = set()
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.grid[y][x] == 0: continue
                # Horizontal
                if x < self.BOARD_WIDTH - 2 and self.grid[y][x] == self.grid[y][x + 1] == self.grid[y][x + 2]:
                    matches.add((x, y)); matches.add((x + 1, y)); matches.add((x + 2, y))
                # Vertical
                if y < self.BOARD_HEIGHT - 2 and self.grid[y][x] == self.grid[y + 1][x] == self.grid[y + 2][x]:
                    matches.add((x, y)); matches.add((x, y + 1)); matches.add((x, y + 2))
        return matches

    def _handle_matches(self, matches):
        reward = 0
        for x, y in matches:
            if self.grid[y][x] != 0:
                self._create_clear_animation((x, y))
                self._create_particles((x, y), self.grid[y][x])
                self.grid[y][x] = 0
                self.score += 10
                self.gems_cleared += 1
                reward += 1
        if len(matches) >= 4:
            self.score += (len(matches) - 3) * 10  # Bonus for longer matches
        return reward

    def _apply_gravity_and_refill(self):
        for x in range(self.BOARD_WIDTH):
            empty_count = 0
            for y in range(self.BOARD_HEIGHT - 1, -1, -1):
                if self.grid[y][x] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    gem_type = self.grid[y][x]
                    self.grid[y + empty_count][x] = gem_type
                    self.grid[y][x] = 0
                    self._create_fall_animation((x, y), (x, y + empty_count), gem_type)

            # Refill from top
            for i in range(empty_count):
                gem_type = self.np_random.integers(1, self.GEM_TYPES + 1)
                self.grid[i][x] = gem_type
                self._create_fall_animation((x, i - empty_count), (x, i), gem_type)

    def _populate_board(self):
        self.grid = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=int)
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                possible_gems = list(range(1, self.GEM_TYPES + 1))
                # Avoid creating matches on spawn
                if x > 1 and self.grid[y][x - 1] == self.grid[y][x - 2] and self.grid[y][x - 1] in possible_gems:
                    possible_gems.remove(self.grid[y][x - 1])
                if y > 1 and self.grid[y - 1][x] == self.grid[y - 2][x] and self.grid[y - 1][x] in possible_gems:
                    possible_gems.remove(self.grid[y - 1][x])
                self.grid[y][x] = self.np_random.choice(possible_gems)

    def _get_observation(self):
        # --- Background and Grid ---
        self.screen.fill(self.COLOR_BG)
        for y in range(self.BOARD_HEIGHT + 1):
            start_pos = (self.GRID_START_X, self.GRID_START_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_START_X + self.BOARD_WIDTH * self.CELL_SIZE, self.GRID_START_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for x in range(self.BOARD_WIDTH + 1):
            start_pos = (self.GRID_START_X + x * self.CELL_SIZE, self.GRID_START_Y)
            end_pos = (self.GRID_START_X + x * self.CELL_SIZE, self.GRID_START_Y + self.BOARD_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # --- Draw Gems (Static) ---
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                gem_type = self.grid[y][x]
                if gem_type > 0:
                    is_animated = any(
                        (anim['type'] in ['swap', 'rewind_swap'] and ((x, y) == anim['p1'] or (x, y) == anim['p2'])) or
                        (anim['type'] == 'clear' and (x, y) == anim['pos']) or
                        (anim['type'] == 'fall' and (x, y) == anim['to_pos'])
                        for anim in self.animations
                    )
                    if not is_animated:
                        self._draw_gem(self.screen, (x, y), gem_type)

        # --- Draw Animated Objects ---
        self._draw_animations()
        self._draw_particles()

        # --- Draw Cursor and Selection ---
        if self.game_state == 'INPUT' and not self.game_over:
            cx, cy = self.cursor_pos
            rect = (self.GRID_START_X + cx * self.CELL_SIZE, self.GRID_START_Y + cy * self.CELL_SIZE, self.CELL_SIZE,
                    self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)
            if self.selected_gem:
                sx, sy = self.selected_gem
                s_rect = (self.GRID_START_X + sx * self.CELL_SIZE + 2, self.GRID_START_Y + sy * self.CELL_SIZE + 2,
                          self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                pygame.draw.rect(self.screen, self.COLOR_SELECTED, s_rect, 3)

        # --- UI Overlay ---
        self._render_ui()

        # --- Game Over Screen ---
        if self.game_over:
            self._render_game_over()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, pos):
        x, y = pos
        return (
            self.GRID_START_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_START_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _draw_gem(self, surface, pos, gem_type, scale=1.0):
        px_float, py_float = self._grid_to_pixel(pos)
        px, py = int(px_float), int(py_float)

        color = self.GEM_COLORS[gem_type - 1]
        radius = int(self.GEM_SIZE * scale)
        if radius <= 0: return

        # Main gem body
        pygame.gfxdraw.aacircle(surface, px, py, radius, color)
        pygame.gfxdraw.filled_circle(surface, px, py, radius, color)
        # Highlight
        highlight_color = tuple(min(255, c + 80) for c in color)
        pygame.gfxdraw.aacircle(surface, px - radius // 3, py - radius // 3, radius // 3, highlight_color)
        pygame.gfxdraw.filled_circle(surface, px - radius // 3, py - radius // 3, radius // 3, highlight_color)

    def _draw_animations(self):
        for anim in self.animations:
            progress_ratio = anim['progress'] / anim['duration']
            if anim['type'] == 'swap' or anim['type'] == 'rewind_swap':
                p1_x, p1_y = anim['p1']
                p2_x, p2_y = anim['p2']

                # Interpolate grid coordinates
                g1_curr_x = p1_x + (p2_x - p1_x) * progress_ratio
                g1_curr_y = p1_y + (p2_y - p1_y) * progress_ratio
                g2_curr_x = p2_x + (p1_x - p2_x) * progress_ratio
                g2_curr_y = p2_y + (p1_y - p2_y) * progress_ratio

                self._draw_gem(self.screen, (g1_curr_x, g1_curr_y), anim['g1_type'])
                self._draw_gem(self.screen, (g2_curr_x, g2_curr_y), anim['g2_type'])

            elif anim['type'] == 'clear':
                scale = 1.0 - progress_ratio
                self._draw_gem(self.screen, anim['pos'], anim['gem_type'], scale=scale)

            elif anim['type'] == 'fall':
                from_x, from_y = anim['from_pos']
                to_x, to_y = anim['to_pos']

                # Interpolate grid coordinates
                curr_x = from_x + (to_x - from_x) * progress_ratio
                curr_y = from_y + (to_y - from_y) * progress_ratio

                self._draw_gem(self.screen, (curr_x, curr_y), anim['gem_type'])

    def _draw_particles(self):
        survivors = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                survivors.append(p)
                size = max(1, int(p['size'] * (p['life'] / p['max_life'])))
                rect = (int(p['pos'][0]), int(p['pos'][1]), size, size)
                pygame.draw.rect(self.screen, p['color'], rect)
        self.particles = survivors

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 10))

        gems_text = self.font_small.render(f"Gems: {self.gems_cleared}/{self.WIN_GEM_TARGET}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (20, 40))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))

        if self.win_status == 'win':
            text = "YOU WIN!"
            color = (100, 255, 100)
        else:
            text = "GAME OVER"
            color = (255, 100, 100)

        game_over_surf = self.font_game_over.render(text, True, color)
        pos = (self.WIDTH // 2 - game_over_surf.get_width() // 2, self.HEIGHT // 2 - game_over_surf.get_height() // 2)
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(game_over_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "gems_cleared": self.gems_cleared,
            "game_state": self.game_state
        }

    # --- Animation Creation ---
    def _create_swap_animation(self, p1, p2, is_rewind=False):
        g1_type = self.grid[p1[1]][p1[0]]
        g2_type = self.grid[p2[1]][p2[0]]
        self.animations.append({
            'type': 'rewind_swap' if is_rewind else 'swap',
            'p1': p1, 'p2': p2, 'g1_type': g1_type, 'g2_type': g2_type,
            'progress': 0, 'duration': 10
        })

    def _create_clear_animation(self, pos):
        self.animations.append({
            'type': 'clear', 'pos': pos, 'gem_type': self.grid[pos[1]][pos[0]],
            'progress': 0, 'duration': 15
        })

    def _create_fall_animation(self, from_pos, to_pos, gem_type):
        dist = to_pos[1] - from_pos[1]
        self.animations.append({
            'type': 'fall', 'from_pos': from_pos, 'to_pos': to_pos, 'gem_type': gem_type,
            'progress': 0, 'duration': int(math.sqrt(dist) * 4) + 2
        })

    def _create_particles(self, pos, gem_type):
        px, py = self._grid_to_pixel(pos)
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'max_life': 30,
                'color': color,
                'size': random.randint(2, 5)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Use a dummy window to display the game
    pygame.display.set_caption("Gem Puzzle")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False
    running = True

    # Game loop
    while running:
        action = [0, 0, 0]  # Default no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            # Movement
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            else: action[0] = 0

            # Space
            if keys[pygame.K_SPACE]: action[1] = 1

            # Shift
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)  # Control the frame rate

        if terminated and running:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            pygame.time.wait(3000)  # Pause for 3 seconds
            obs, info = env.reset()
            terminated = False

    pygame.quit()