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
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, then press Space on an adjacent tile to swap them. Match 3 or more to score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent gems to create lines of three or more. Clear the entire board before the timer runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.NUM_COLORS = 5
        self.MAX_STEPS = 60 * 30  # 60 seconds at 30fps

        self.GRID_WIDTH = 240
        self.GRID_HEIGHT = 240
        self.TILE_SIZE = self.GRID_WIDTH // self.GRID_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SCORE = (255, 215, 0)
        self.COLOR_TIME = (0, 200, 255)
        self.TILE_COLORS = [
            (255, 80, 80),  # Red
            (80, 255, 80),  # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_tile = None
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.game_state = 'IDLE'
        self.animations = []
        self.last_space_held = False
        self.total_tiles = self.GRID_SIZE * self.GRID_SIZE
        self.tiles_cleared_total = 0

        # Initialize state variables
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.selected_tile = None
        self.game_state = 'IDLE'
        self.animations = []
        self.last_space_held = False
        self.tiles_cleared_total = 0

        self._create_board()

        return self._get_observation(), self._get_info()

    def _create_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_matches_on_board() and self._find_possible_moves():
                break

    def step(self, action):
        reward = -0.01  # Time penalty
        self.steps += 1
        self.time_left -= 1

        movement, space_held, _ = int(action[0]), action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        if self.game_state == 'IDLE':
            self._handle_input(movement, space_pressed)

        reward += self._update_game_state()

        terminated = self.game_over
        if not terminated:
            if self.time_left <= 0:
                reward -= 100  # Terminal penalty for time out
                terminated = True
            elif self.tiles_cleared_total >= self.total_tiles:
                reward += 100  # Terminal reward for clearing board
                terminated = True

        self.game_over = terminated
        truncated = False # This environment does not truncate based on time limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Move cursor
        cx, cy = self.cursor_pos
        if movement == 1: self.cursor_pos = (cx, (cy - 1 + self.GRID_SIZE) % self.GRID_SIZE)
        elif movement == 2: self.cursor_pos = (cx, (cy + 1) % self.GRID_SIZE)
        elif movement == 3: self.cursor_pos = ((cx - 1 + self.GRID_SIZE) % self.GRID_SIZE, cy)
        elif movement == 4: self.cursor_pos = ((cx + 1) % self.GRID_SIZE, cy)

        if space_pressed:
            # sound: select_gem.wav
            if self.selected_tile is None:
                self.selected_tile = self.cursor_pos
            else:
                sx, sy = self.selected_tile
                cx, cy = self.cursor_pos
                if sx == cx and sy == cy:  # Deselect
                    self.selected_tile = None
                elif abs(sx - cx) + abs(sy - cy) == 1:  # Adjacent, trigger swap
                    self._start_swap(self.selected_tile, self.cursor_pos)
                    self.selected_tile = None
                else:  # Select new tile
                    self.selected_tile = self.cursor_pos
        elif movement == 0 and not space_pressed and self.selected_tile is not None:
             # Deselect on no-op without space press, to allow moving away from selection
             self.selected_tile = None


    def _update_game_state(self):
        # Process animations
        if self.animations:
            for anim in self.animations:
                anim['progress'] += 1
            self.animations = [anim for anim in self.animations if anim['progress'] < anim['duration']]
            if self.animations:
                return 0  # Still animating, no new state changes or rewards

        # State machine transitions after animations complete
        if self.game_state == 'SWAPPING':
            pos1, pos2 = self.swap_info['pos1'], self.swap_info['pos2']
            self.grid[pos1[1], pos1[0]], self.grid[pos2[1], pos2[0]] = self.grid[pos2[1], pos2[0]], self.grid[pos1[1], pos1[0]]
            matches = self._find_matches_on_board()
            if matches:
                # sound: match_success.wav
                return self._start_clearing(matches)
            else:
                # sound: invalid_swap.wav
                self._start_swap(pos1, pos2, is_return=True)  # Swap back
                return 0

        elif self.game_state == 'CLEARING':
            return self._start_gravity()

        elif self.game_state == 'REFILLING':
            matches = self._find_matches_on_board()
            if matches:
                # sound: chain_reaction.wav
                return self._start_clearing(matches)  # Chain reaction
            else:
                if not self._find_possible_moves():
                    # sound: reshuffle.wav
                    self._start_reshuffle()
                else:
                    self.game_state = 'IDLE'

        elif self.game_state == 'RESHUFFLING':
            self._create_board()
            self.game_state = 'IDLE'

        return 0

    def _start_swap(self, pos1, pos2, is_return=False):
        self.game_state = 'SWAPPING' if not is_return else 'IDLE'
        self.swap_info = {'pos1': pos1, 'pos2': pos2}
        duration = 8
        self.animations.append({'type': 'swap', 'pos': pos1, 'target': pos2, 'duration': duration, 'progress': 0})
        self.animations.append({'type': 'swap', 'pos': pos2, 'target': pos1, 'duration': duration, 'progress': 0})

    def _start_clearing(self, matches):
        self.game_state = 'CLEARING'
        duration = 10
        reward = 0
        num_cleared = len(matches)

        reward += num_cleared  # +1 per tile
        if num_cleared == 4: reward += 5
        if num_cleared >= 5: reward += 10

        self.score += reward
        self.tiles_cleared_total += num_cleared

        for x, y in matches:
            self.animations.append({'type': 'clear', 'pos': (x, y), 'duration': duration, 'progress': 0})
            self.grid[y, x] = -1  # Mark as empty
        return reward

    def _start_gravity(self):
        self.game_state = 'REFILLING'
        duration = 12

        for x in range(self.GRID_SIZE):
            empty_count = 0
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    # Animate tile falling
                    self.animations.append({'type': 'fall', 'pos': (x, y), 'target_y': y + empty_count, 'duration': duration, 'progress': 0})
                    self.grid[y + empty_count, x] = self.grid[y, x]
                    self.grid[y, x] = -1

        # Refill new tiles from top
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.grid[y, x] == -1:
                    new_color = self.np_random.integers(0, self.NUM_COLORS)
                    self.grid[y, x] = new_color
                    self.animations.append({'type': 'new_tile', 'pos': (x, y), 'start_y': y - self.GRID_SIZE, 'duration': duration, 'progress': 0})
        return 0

    def _start_reshuffle(self):
        self.game_state = 'RESHUFFLING'
        duration = 30
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                self.animations.append({'type': 'shuffle', 'pos': (x, y), 'duration': duration, 'progress': 0})

    def _find_matches_on_board(self):
        matches = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y, x] == -1: continue
                # Horizontal
                if x < self.GRID_SIZE - 2 and self.grid[y, x] == self.grid[y, x + 1] == self.grid[y, x + 2]:
                    matches.update([(x, y), (x + 1, y), (x + 2, y)])
                # Vertical
                if y < self.GRID_SIZE - 2 and self.grid[y, x] == self.grid[y + 1, x] == self.grid[y + 2, x]:
                    matches.update([(x, y), (x, y + 1), (x, y + 2)])
        return list(matches)

    def _find_possible_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Try swapping right
                if x < self.GRID_SIZE - 1:
                    self.grid[y, x], self.grid[y, x + 1] = self.grid[y, x + 1], self.grid[y, x]
                    if self._find_matches_on_board():
                        self.grid[y, x], self.grid[y, x + 1] = self.grid[y, x + 1], self.grid[y, x]  # Swap back
                        return True
                    self.grid[y, x], self.grid[y, x + 1] = self.grid[y, x + 1], self.grid[y, x]  # Swap back
                # Try swapping down
                if y < self.GRID_SIZE - 1:
                    self.grid[y, x], self.grid[y + 1, x] = self.grid[y + 1, x], self.grid[y, x]
                    if self._find_matches_on_board():
                        self.grid[y, x], self.grid[y + 1, x] = self.grid[y + 1, x], self.grid[y, x]  # Swap back
                        return True
                    self.grid[y, x], self.grid[y + 1, x] = self.grid[y + 1, x], self.grid[y, x]  # Swap back
        return False

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
            "time_left": self.time_left,
            "tiles_cleared": self.tiles_cleared_total,
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw tiles
        rendered_tiles = set()
        for anim in self.animations:
            if anim['type'] in ['swap', 'fall', 'new_tile']:
                pos = anim['pos']
                rendered_tiles.add(pos)
                self._draw_animated_tile(anim)

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) not in rendered_tiles and self.grid[y, x] != -1:
                    is_clearing = any(anim['type'] == 'clear' and anim['pos'] == (x, y) for anim in self.animations)
                    is_shuffling = any(anim['type'] == 'shuffle' and anim['pos'] == (x, y) for anim in self.animations)
                    if is_clearing:
                        anim = next(anim for anim in self.animations if anim['type'] == 'clear' and anim['pos'] == (x, y))
                        self._draw_tile(x, y, self.grid[y, x], scale_anim=anim)
                    elif is_shuffling:
                        anim = next(anim for anim in self.animations if anim['type'] == 'shuffle' and anim['pos'] == (x, y))
                        self._draw_tile(x, y, self.grid[y, x], shuffle_anim=anim)
                    else:
                        self._draw_tile(x, y, self.grid[y, x])

        # Draw cursor and selection
        self._draw_cursor()

    def _draw_animated_tile(self, anim):
        progress = anim['progress'] / anim['duration']

        if anim['type'] == 'swap':
            start_pos = anim['pos']
            end_pos = anim['target']
            interp_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            interp_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            self._draw_tile(interp_x, interp_y, self.grid[start_pos[1], start_pos[0]])
        elif anim['type'] == 'fall':
            x, start_y = anim['pos']
            end_y = anim['target_y']
            interp_y = start_y + (end_y - start_y) * progress
            self._draw_tile(x, interp_y, self.grid[end_y, x])
        elif anim['type'] == 'new_tile':
            x, end_y = anim['pos']
            start_y = anim['start_y']
            interp_y = start_y + (end_y - start_y) * progress
            self._draw_tile(x, interp_y, self.grid[end_y, x])

    def _draw_tile(self, x, y, color_idx, scale_anim=None, shuffle_anim=None):
        if color_idx < 0: return

        px = self.GRID_X + x * self.TILE_SIZE
        py = self.GRID_Y + y * self.TILE_SIZE

        tile_rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
        color = self.TILE_COLORS[color_idx]

        if scale_anim:
            progress = scale_anim['progress'] / scale_anim['duration']
            scale = 1.0 - progress
            center = tile_rect.center
            tile_rect.width = int(self.TILE_SIZE * scale)
            tile_rect.height = int(self.TILE_SIZE * scale)
            tile_rect.center = center

        if shuffle_anim:
            progress = shuffle_anim['progress'] / shuffle_anim['duration']
            angle = progress * 360
            scale = 1.0 - abs(0.5 - progress) * 2.0
            center = tile_rect.center
            tile_rect.width = int(self.TILE_SIZE * scale)
            tile_rect.height = int(self.TILE_SIZE * scale)
            tile_rect.center = center

        pygame.draw.rect(self.screen, color, tile_rect, border_radius=5)

    def _draw_cursor(self):
        # Draw selected tile highlight
        if self.selected_tile:
            sx, sy = self.selected_tile
            rect = pygame.Rect(self.GRID_X + sx * self.TILE_SIZE, self.GRID_Y + sy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(4, 4), 3, border_radius=7)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(self.GRID_X + cx * self.TILE_SIZE, self.GRID_Y + cy * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pulse = abs(math.sin(self.steps * 0.2))
        color = (200 + 55 * pulse, 200 + 55 * pulse, 255)
        pygame.draw.rect(self.screen, color, rect.inflate(2, 2), 2, border_radius=6)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(center=(self.WIDTH // 4, 40))
        self.screen.blit(score_text, score_rect)

        # Time
        time_percent = max(0, self.time_left / self.MAX_STEPS)
        time_bar_width = 200
        time_bar_rect = pygame.Rect(0, 0, int(time_bar_width * time_percent), 20)
        time_bar_rect.center = (self.WIDTH // 2, 40)

        # Animate color from green to red
        bar_color = (255 * (1 - time_percent), 200 * time_percent, 50)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (time_bar_rect.x - 2, time_bar_rect.y - 2, time_bar_width + 4, 24), border_radius=5)
        pygame.draw.rect(self.screen, bar_color, time_bar_rect, border_radius=5)

        # Tiles remaining
        tiles_left = max(0, self.total_tiles - self.tiles_cleared_total)
        tiles_text = self.font_medium.render(f"Tiles: {tiles_left}", True, self.COLOR_TEXT)
        tiles_rect = tiles_text.get_rect(center=(self.WIDTH * 3 // 4, 40))
        self.screen.blit(tiles_text, tiles_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.tiles_cleared_total >= self.total_tiles:
                msg = "BOARD CLEARED!"
            else:
                msg = "TIME UP!"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()

    # --- To play with keyboard ---
    # This part requires a display
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Match-3 Gym Environment")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        # Re-enable display for interactive mode
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")

        while not done:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0  # none
            
            # This event loop is needed to register key presses properly
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                    if event.key == pygame.K_UP:
                        movement = 1
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                    elif event.key == pygame.K_LEFT:
                        movement = 3
                    elif event.key == pygame.K_RIGHT:
                        movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = np.array([movement, space_held, shift_held])

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))

            pygame.display.flip()

            clock.tick(30)  # Match the environment's intended FPS

        pygame.quit()
    except pygame.error:
        print("Pygame display could not be initialized. Running a headless test.")
        obs, info = env.reset()
        done = False
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']}")
                obs, info = env.reset()
        print("Headless test complete.")