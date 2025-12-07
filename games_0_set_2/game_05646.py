import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to swap the selected fruit "
        "with the one in the direction of your last move. Match 3 or more to score!"
    )

    game_description = (
        "Fast-paced arcade puzzle. Match cascading fruits in a grid to create combos and "
        "reach the target score before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    GRID_COLS, GRID_ROWS = 8, 8
    NUM_FRUIT_TYPES = 5
    CELL_SIZE = 50
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE

    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    TARGET_SCORE = 100
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = 1800  # 60 seconds * 30 fps

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (40, 55, 70)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SCORE = (255, 220, 0)
    COLOR_TIMER_NORMAL = (0, 200, 255)
    COLOR_TIMER_WARN = (255, 80, 80)
    COLOR_CURSOR = (255, 255, 255)
    FRUIT_COLORS = [
        (255, 80, 80),  # Red
        (80, 255, 80),  # Green
        (80, 120, 255),  # Blue
        (255, 220, 0),  # Yellow
        (200, 80, 255),  # Purple
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_combo = pygame.font.Font(None, 48)
        self.font_game_over = pygame.font.Font(None, 72)

        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.cursor_pos = [0, 0]
        self.last_move_dir = [1, 0]  # Start by pointing right
        self.prev_space_held = False
        self.animations = []
        self.particles = []
        self.combo_meter = 0
        self.combo_fade_timer = 0.0

        self.np_random = None

        # self.reset() is called by the wrapper, but good practice to have a defined state
        self._initialize_state()

    def _initialize_state(self):
        self.steps = 0
        self.score = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.just_swapped = False

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.last_move_dir = [1, 0]
        self.prev_space_held = False
        self.animations.clear()
        self.particles.clear()
        self.combo_meter = 0
        self.combo_fade_timer = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._initialize_state()
        self._generate_and_validate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            dt = self.clock.tick(30) / 1000.0
        else:
            dt = 1 / 30.0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.time_remaining = max(0, self.time_remaining - dt)
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        is_animating = any(anim['type'] != 'pop' for anim in self.animations)

        if not is_animating:
            self._handle_input(movement, space_held)
            if self.just_swapped:
                reward += self._process_swap()
                self.just_swapped = False

        self._update_animations(dt)
        self._update_particles(dt)

        if self.combo_fade_timer > 0:
            self.combo_fade_timer -= dt
        else:
            self.combo_meter = 0

        terminated = self._check_termination()
        if terminated:
            if self.score >= self.TARGET_SCORE:
                reward += 100.0  # Win bonus
            else:
                reward += -50.0  # Loss penalty
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        self.just_swapped = False
        moved = False
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            self.last_move_dir = [0, -1]
            moved = True
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            self.last_move_dir = [0, 1]
            moved = True
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            self.last_move_dir = [-1, 0]
            moved = True
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
            self.last_move_dir = [1, 0]
            moved = True

        if moved:
            # sfx: cursor_move.wav
            pass

        if space_held and not self.prev_space_held:
            self.just_swapped = True
            # sfx: swap_attempt.wav

        self.prev_space_held = space_held

    def _process_swap(self):
        c1, r1 = self.cursor_pos
        c2, r2 = c1 + self.last_move_dir[0], r1 + self.last_move_dir[1]

        if not (0 <= c2 < self.GRID_COLS and 0 <= r2 < self.GRID_ROWS):
            return 0  # Invalid swap location

        # Perform swap
        self.grid[c1, r1], self.grid[c2, r2] = self.grid[c2, r2], self.grid[c1, r1]
        self._add_animation('swap', (c1, r1), (c2, r2))

        matches = self._find_all_matches()
        if not matches:
            # Invalid swap, swap back
            self.grid[c1, r1], self.grid[c2, r2] = self.grid[c2, r2], self.grid[c1, r1]
            self._add_animation('swap', (c1, r1), (c2, r2), delay=0.25)
            # sfx: invalid_swap.wav
            return -0.2

        # Valid swap, process chain reaction
        total_reward = 0
        self.combo_meter = 0

        while matches:
            self.combo_meter += 1
            self.combo_fade_timer = 1.5  # seconds

            num_matched = len(matches)
            total_reward += num_matched  # +1 per fruit

            if num_matched >= 4: total_reward += 5
            if num_matched >= 5: total_reward += 5  # Additional bonus for 5+

            # sfx: match_{combo_meter}.wav

            for (c, r) in matches:
                self._add_animation('pop', (c, r), fruit_type=self.grid[c, r])
                self.grid[c, r] = -1  # Mark for removal

            self._apply_gravity()
            self._refill_grid()
            matches = self._find_all_matches()

        # After a successful chain, ensure board is still playable
        if not self._has_possible_moves():
            self._generate_and_validate_grid()

        return total_reward

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[c, r] != -1 and self.grid[c, r] == self.grid[c + 1, r] == self.grid[c + 2, r]:
                    matches.update([(c, r), (c + 1, r), (c + 2, r)])
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[c, r] != -1 and self.grid[c, r] == self.grid[c, r + 1] == self.grid[c, r + 2]:
                    matches.update([(c, r), (c, r + 1), (c, r + 2)])
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            write_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[c, r] != -1:
                    if r != write_row:
                        self.grid[c, write_row] = self.grid[c, r]
                        self._add_animation('fall', (c, r), (c, write_row), fruit_type=self.grid[c, write_row])
                        self.grid[c, r] = -1
                    write_row -= 1

    def _refill_grid(self):
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[c, r] == -1:
                    self.grid[c, r] = self.np_random.integers(0, self.NUM_FRUIT_TYPES)
                    self._add_animation('fall', (c, -1), (c, r), fruit_type=self.grid[c, r])

    def _generate_and_validate_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_FRUIT_TYPES, size=(self.GRID_COLS, self.GRID_ROWS))
            if not self._find_all_matches() and self._has_possible_moves():
                break

    def _has_possible_moves(self):
        temp_grid = np.copy(self.grid)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    temp_grid[c, r], temp_grid[c + 1, r] = temp_grid[c + 1, r], temp_grid[c, r]
                    if self._check_matches_at(temp_grid, c, r) or self._check_matches_at(temp_grid, c + 1, r):
                        return True
                    temp_grid[c, r], temp_grid[c + 1, r] = temp_grid[c + 1, r], temp_grid[c, r]  # Swap back
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    temp_grid[c, r], temp_grid[c, r + 1] = temp_grid[c, r + 1], temp_grid[c, r]
                    if self._check_matches_at(temp_grid, c, r) or self._check_matches_at(temp_grid, c, r + 1):
                        return True
                    temp_grid[c, r], temp_grid[c, r + 1] = temp_grid[c, r + 1], temp_grid[c, r]  # Swap back
        return False

    def _check_matches_at(self, grid, c, r):
        fruit = grid[c, r]
        # Horizontal
        h_count = 1
        for i in range(1, 3):
            if c - i >= 0 and grid[c - i, r] == fruit:
                h_count += 1
            else:
                break
        for i in range(1, 3):
            if c + i < self.GRID_COLS and grid[c + i, r] == fruit:
                h_count += 1
            else:
                break
        if h_count >= 3: return True
        # Vertical
        v_count = 1
        for i in range(1, 3):
            if r - i >= 0 and grid[c, r - i] == fruit:
                v_count += 1
            else:
                break
        for i in range(1, 3):
            if r + i < self.GRID_ROWS and grid[c, r + i] == fruit:
                v_count += 1
            else:
                break
        if v_count >= 3: return True
        return False

    def _check_termination(self):
        return self.time_remaining <= 0 or self.score >= self.TARGET_SCORE or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time": self.time_remaining}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=10)

        # Draw fruits
        visible_fruits = set((c, r) for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS))

        # Draw animated fruits first
        for anim in self.animations:
            if anim['type'] in ['swap', 'fall']:
                p = anim['progress']

                if anim['type'] == 'swap':
                    c1, r1 = anim['pos1']
                    c2, r2 = anim['pos2']
                    x1, y1 = self._grid_to_pixel(c1, r1)
                    x2, y2 = self._grid_to_pixel(c2, r2)

                    # Draw fruit 1 moving to pos 2
                    px1 = x1 + (x2 - x1) * p
                    py1 = y1 + (y2 - y1) * p
                    self._draw_fruit(px1, py1, self.grid[c2, r2])

                    # Draw fruit 2 moving to pos 1
                    px2 = x2 + (x1 - x2) * p
                    py2 = y2 + (y1 - y2) * p
                    self._draw_fruit(px2, py2, self.grid[c1, r1])

                    visible_fruits.discard((c1, r1))
                    visible_fruits.discard((c2, r2))

                elif anim['type'] == 'fall':
                    c1, r1 = anim['pos1']
                    c2, r2 = anim['pos2']
                    x1, y1 = self._grid_to_pixel(c1, r1)
                    x2, y2 = self._grid_to_pixel(c2, r2)

                    px = x1 + (x2 - x1) * p
                    py = y1 + (y2 - y1) * p
                    self._draw_fruit(px, py, anim['fruit_type'])
                    visible_fruits.discard((c2, r2))

            elif anim['type'] == 'pop':
                p = anim['progress']
                c, r = anim['pos1']  # FIX: Changed 'pos' to 'pos1'
                x, y = self._grid_to_pixel(c, r)
                radius = int(self.CELL_SIZE * 0.4 * (1 - p))
                alpha = int(255 * (1 - p))
                if radius > 0:
                    self._draw_fruit(x, y, anim['fruit_type'], radius, alpha)
                visible_fruits.discard((c, r))

        # Draw static fruits
        for c, r in visible_fruits:
            fruit_type = self.grid[c, r]
            if fruit_type != -1:
                x, y = self._grid_to_pixel(c, r)
                self._draw_fruit(x, y, fruit_type)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

        # Draw cursor
        if not self.game_over:
            cx, cy = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
            cursor_rect = pygame.Rect(cx - self.CELL_SIZE // 2, cy - self.CELL_SIZE // 2, self.CELL_SIZE,
                                      self.CELL_SIZE)

            # Pulsing effect
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
            alpha = 150 + pulse * 105

            # Draw main cursor
            pygame.gfxdraw.rectangle(self.screen, cursor_rect, (*self.COLOR_CURSOR, alpha))

            # Draw adjacent selection indicator
            ac1, ar1 = self.cursor_pos
            ac2, ar2 = ac1 + self.last_move_dir[0], ar1 + self.last_move_dir[1]
            if 0 <= ac2 < self.GRID_COLS and 0 <= ar2 < self.GRID_ROWS:
                ax, ay = self._grid_to_pixel(ac2, ar2)
                adj_rect = pygame.Rect(ax - self.CELL_SIZE // 2, ay - self.CELL_SIZE // 2, self.CELL_SIZE,
                                       self.CELL_SIZE)
                pygame.gfxdraw.rectangle(self.screen, adj_rect, (*self.COLOR_CURSOR, alpha // 3))

    def _draw_fruit(self, x, y, fruit_type, radius=None, alpha=255):
        if radius is None:
            radius = int(self.CELL_SIZE * 0.4)

        color = self.FRUIT_COLORS[fruit_type]

        # Main body with alpha
        target_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, (*color, alpha), (radius, radius), radius)

        # Highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.draw.circle(shape_surf, (*highlight_color, alpha), (int(radius * 0.7), int(radius * 0.7)),
                           int(radius * 0.3))

        self.screen.blit(shape_surf, target_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))

        # Timer
        timer_color = self.COLOR_TIMER_NORMAL if self.time_remaining > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_main.render(f"Time: {int(self.time_remaining)}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 10))

        # Combo Meter
        if self.combo_meter > 1 and self.combo_fade_timer > 0:
            fade_alpha = min(255, int(255 * (self.combo_fade_timer / 1.0)))
            combo_surf = self.font_combo.render(f"x{self.combo_meter} Combo!", True, self.COLOR_SCORE)
            combo_surf.set_alpha(fade_alpha)
            pos_x = self.SCREEN_WIDTH / 2 - combo_surf.get_width() / 2
            pos_y = self.SCREEN_HEIGHT / 2 - combo_surf.get_height() / 2
            self.screen.blit(combo_surf, (pos_x, pos_y))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg = "You Win!" if self.score >= self.TARGET_SCORE else "Time's Up!"
            game_over_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _grid_to_pixel(self, c, r):
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _add_animation(self, anim_type, pos1, pos2=None, fruit_type=None, delay=0.0):
        duration = {'swap': 0.25, 'fall': 0.2, 'pop': 0.3}.get(anim_type, 0)
        self.animations.append({
            'type': anim_type, 'pos1': pos1, 'pos2': pos2, 'fruit_type': fruit_type,
            'duration': duration, 'progress': 0.0, 'delay': delay
        })
        if anim_type == 'pop':
            self._create_particles(pos1, fruit_type)

    def _update_animations(self, dt):
        for anim in self.animations[:]:
            if anim['delay'] > 0:
                anim['delay'] -= dt
                continue

            anim['progress'] += dt / anim['duration']
            if anim['progress'] >= 1.0:
                self.animations.remove(anim)

    def _create_particles(self, pos, fruit_type):
        px, py = self._grid_to_pixel(pos[0], pos[1])
        color = self.FRUIT_COLORS[fruit_type]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': random.uniform(2, 5),
                'lifespan': random.uniform(0.3, 0.7),
                'color': color
            })

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            p['lifespan'] -= dt
            p['size'] -= 2 * dt
            if p['lifespan'] <= 0 or p['size'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Matcher")

    running = True
    total_reward = 0

    action = np.zeros(env.action_space.shape)

    while running:
        # --- Human Input ---
        movement = 0
        space_held = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1

        action = np.array([movement, space_held, 0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False

    env.close()