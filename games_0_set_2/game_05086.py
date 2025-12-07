import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move selector. Space to swap tile with the one in the last moved direction."
    )

    game_description = (
        "Swap adjacent colored tiles to create matches of 3 or more. Clear the board before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    BOARD_WIDTH, BOARD_HEIGHT = 8, 8
    TILE_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - BOARD_WIDTH * TILE_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - BOARD_HEIGHT * TILE_SIZE) // 2 + 20
    ANIMATION_SPEED = 0.25  # Progress per frame
    MAX_MOVES = 50
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (50, 50, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 255, 0)

    TILE_COLORS = {
        1: (220, 50, 50),   # Red
        2: (50, 220, 50),   # Green
        3: (50, 100, 220),  # Blue
        4: (220, 220, 50),  # Yellow
        5: (200, 50, 220),  # Purple
        6: (100, 100, 110)  # Obstacle
    }
    TILE_EMPTY = 0
    TILE_OBSTACLE = 6
    NUM_TILE_TYPES = len(TILE_COLORS) - 1 # Exclude obstacle

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
        self.font_main = pygame.font.SysFont("tahoma", 24, bold=True)
        self.font_small = pygame.font.SysFont("tahoma", 18)

        self.board = None
        self.selector_pos = None
        self.last_move_dir = None
        self.moves_remaining = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.animations = []
        self.particles = []
        self.current_action_reward = 0

        # self.reset() is called by the validation function in this setup
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.selector_pos = [self.BOARD_WIDTH // 2, self.BOARD_HEIGHT // 2]
        self.last_move_dir = [1, 0]  # Default to right
        self.animations = []
        self.particles = []
        self.current_action_reward = 0

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.current_action_reward = 0

        # Game logic only proceeds if no animations are running
        if not self.animations:
            self._handle_input(action)

        # Update animations and game state progression (falls, clears, etc.)
        self._update_animations()

        # Check for termination conditions
        terminated = self._check_termination()
        reward = self.current_action_reward

        if terminated and not self.game_over:
            self.game_over = True
            # Apply terminal rewards
            is_win = self._check_win_condition()
            reward += 100 if is_win else -50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action

        # --- Selector Movement ---
        moved = False
        if movement == 1: # Up
            self.selector_pos[1] -= 1
            self.last_move_dir = [0, -1]
            moved = True
        elif movement == 2: # Down
            self.selector_pos[1] += 1
            self.last_move_dir = [0, 1]
            moved = True
        elif movement == 3: # Left
            self.selector_pos[0] -= 1
            self.last_move_dir = [-1, 0]
            moved = True
        elif movement == 4: # Right
            self.selector_pos[0] += 1
            self.last_move_dir = [1, 0]
            moved = True

        if moved:
            self.selector_pos[0] %= self.BOARD_WIDTH
            self.selector_pos[1] %= self.BOARD_HEIGHT

        # --- Swap Action ---
        if space_pressed:
            self.moves_remaining -= 1

            x1, y1 = self.selector_pos
            x2, y2 = x1 + self.last_move_dir[0], y1 + self.last_move_dir[1]

            # Check if swap is valid
            if not (0 <= x2 < self.BOARD_WIDTH and 0 <= y2 < self.BOARD_HEIGHT) or \
               self.board[y1][x1] == self.TILE_OBSTACLE or \
               self.board[y2][x2] == self.TILE_OBSTACLE:
                self.current_action_reward = -0.1
                # Add invalid move feedback animation (e.g., flash selector red)
                return

            # Temporarily swap to check for matches
            self.board[y1][x1], self.board[y2][x2] = self.board[y2][x2], self.board[y1][x1]
            matches = self._find_matches()

            if not matches:
                # No match, swap back
                self.board[y1][x1], self.board[y2][x2] = self.board[y2][x2], self.board[y1][x1]
                self.current_action_reward = -0.1
                self._add_animation('swap', ((x1, y1), (x2, y2)), on_complete=lambda: self._add_animation('swap', ((x2, y2), (x1, y1))))
                # sfx: invalid_swap_sound
            else:
                # Match found, keep swap and start cascade
                self._add_animation('swap', ((x1, y1), (x2, y2)), on_complete=self._start_cascade)
                # sfx: swap_sound

    def _handle_reshuffle(self):
        self._reshuffle_board()
        self._start_cascade()

    def _start_cascade(self):
        matches = self._find_matches()
        if not matches:
            # After a fall, if no new matches, check for possible moves
            if not self._find_possible_moves():
                self._add_animation('reshuffle', on_complete=self._handle_reshuffle)
            return

        # --- Process Matches ---
        cleared_tiles_count = len(matches)
        self.current_action_reward += cleared_tiles_count * 1.0 # +1 per tile
        if cleared_tiles_count >= 4:
            self.current_action_reward += 5.0 # Bonus for 4+ match
        self.score += cleared_tiles_count * 10

        self._add_animation('clear', matches, on_complete=self._apply_gravity)
        # sfx: match_clear_sound
        for x, y in matches:
            self._create_particles(x, y, self.board[y][x])

        for x, y in matches:
            self.board[y][x] = self.TILE_EMPTY

    def _apply_gravity(self):
        moved_tiles = []
        for x in range(self.BOARD_WIDTH):
            empty_row = self.BOARD_HEIGHT - 1
            for y in range(self.BOARD_HEIGHT - 1, -1, -1):
                if self.board[y][x] != self.TILE_EMPTY:
                    if y != empty_row:
                        self.board[empty_row][x] = self.board[y][x]
                        self.board[y][x] = self.TILE_EMPTY
                        moved_tiles.append(((x, y), (x, empty_row)))
                    empty_row -= 1

        if moved_tiles:
            self._add_animation('fall', moved_tiles, on_complete=self._refill_board)
            # sfx: tiles_fall_sound
        else:
            self._refill_board()

    def _refill_board(self):
        new_tiles = []
        for x in range(self.BOARD_WIDTH):
            for y in range(self.BOARD_HEIGHT):
                if self.board[y][x] == self.TILE_EMPTY:
                    self.board[y][x] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                    new_tiles.append((x, y))

        if new_tiles:
            self._add_animation('refill', new_tiles, on_complete=self._start_cascade)
        else:
            self._start_cascade()

    def _update_animations(self):
        if not self.animations:
            return

        # Process the first animation in the queue
        anim = self.animations[0]
        anim['progress'] = min(1.0, anim['progress'] + self.ANIMATION_SPEED)

        if anim['progress'] >= 1.0:
            self.animations.pop(0)
            if anim['on_complete']:
                anim['on_complete']()

    def _add_animation(self, type, data=None, on_complete=None):
        self.animations.append({
            'type': type,
            'data': data,
            'progress': 0.0,
            'on_complete': on_complete
        })

    def _generate_board(self):
        while True:
            self.board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.BOARD_HEIGHT, self.BOARD_WIDTH))

            # Add some obstacles
            num_obstacles = self.np_random.integers(3, 6)
            for _ in range(num_obstacles):
                x, y = self.np_random.integers(0, self.BOARD_WIDTH), self.np_random.integers(0, self.BOARD_HEIGHT)
                self.board[y][x] = self.TILE_OBSTACLE

            # Ensure no initial matches
            while self._find_matches():
                matches = self._find_matches()
                for x, y in matches:
                    self.board[y][x] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)

            # Ensure at least one move is possible
            if self._find_possible_moves():
                break

    def _find_matches(self):
        matches = set()
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.board[y][x] in [self.TILE_EMPTY, self.TILE_OBSTACLE]:
                    continue
                # Horizontal
                if x < self.BOARD_WIDTH - 2 and self.board[y][x] == self.board[y][x+1] == self.board[y][x+2]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical
                if y < self.BOARD_HEIGHT - 2 and self.board[y][x] == self.board[y+1][x] == self.board[y+2][x]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return list(matches)

    def _find_possible_moves(self):
        moves = []
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                tile = self.board[y][x]
                if tile == self.TILE_OBSTACLE:
                    continue
                # Check swap right
                if x < self.BOARD_WIDTH - 1 and self.board[y][x+1] != self.TILE_OBSTACLE:
                    self.board[y][x], self.board[y][x+1] = self.board[y][x+1], self.board[y][x]
                    if self._find_matches():
                        moves.append(((x, y), (x+1, y)))
                    self.board[y][x], self.board[y][x+1] = self.board[y][x+1], self.board[y][x]
                # Check swap down
                if y < self.BOARD_HEIGHT - 1 and self.board[y+1][x] != self.TILE_OBSTACLE:
                    self.board[y][x], self.board[y+1][x] = self.board[y+1][x], self.board[y][x]
                    if self._find_matches():
                        moves.append(((x, y), (x, y+1)))
                    self.board[y][x], self.board[y+1][x] = self.board[y+1][x], self.board[y][x]
        return moves

    def _reshuffle_board(self):
        movable_tiles = []
        positions = []
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.board[y][x] != self.TILE_OBSTACLE:
                    movable_tiles.append(self.board[y][x])
                    positions.append((x, y))

        while True:
            self.np_random.shuffle(movable_tiles)
            temp_board = self.board.copy()
            for i, (x, y) in enumerate(positions):
                temp_board[y][x] = movable_tiles[i]

            self.board = temp_board
            if not self._find_matches() and self._find_possible_moves():
                break

        self.current_action_reward -= 5 # Penalty for reshuffle
        # sfx: reshuffle_sound

    def _check_win_condition(self):
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.board[y][x] not in [self.TILE_EMPTY, self.TILE_OBSTACLE]:
                    return False
        return True

    def _check_termination(self):
        if self.game_over:
            return True
        if self._check_win_condition():
            return True
        if self.moves_remaining <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()

        # Get tile positions based on animations
        tile_positions = {}
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                tile_positions[(x, y)] = (x, y)

        active_anim = self.animations[0] if self.animations else None
        if active_anim:
            p = active_anim['progress']
            if active_anim['type'] == 'swap':
                (x1, y1), (x2, y2) = active_anim['data']
                tile_positions[(x1, y1)] = (x1 * (1-p) + x2 * p, y1 * (1-p) + y2 * p)
                tile_positions[(x2, y2)] = (x2 * (1-p) + x1 * p, y2 * (1-p) + y1 * p)
            elif active_anim['type'] == 'fall':
                for (x1, y1), (x2, y2) in active_anim['data']:
                    tile_positions[(x2, y2)] = (x1, y1 * (1-p) + y2 * p)

        # Draw tiles
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                tile_val = self.board[y][x]
                if tile_val == self.TILE_EMPTY:
                    continue

                draw_x, draw_y = tile_positions[(x, y)]

                scale = 1.0
                alpha = 255
                if active_anim:
                    if active_anim['type'] == 'clear' and (x, y) in active_anim['data']:
                        scale = 1.0 - active_anim['progress']
                        alpha = 255 * (1.0 - active_anim['progress'])
                    elif active_anim['type'] == 'refill' and (x, y) in active_anim['data']:
                        scale = active_anim['progress']

                self._draw_tile(draw_x, draw_y, tile_val, scale, alpha)

        self._update_and_draw_particles()
        self._draw_selector()

    def _draw_grid(self):
        for y in range(self.BOARD_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            end = (self.GRID_OFFSET_X + self.BOARD_WIDTH * self.TILE_SIZE, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.BOARD_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y + self.BOARD_HEIGHT * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _draw_tile(self, board_x, board_y, tile_val, scale=1.0, alpha=255):
        if tile_val == self.TILE_EMPTY:
            return

        size = int(self.TILE_SIZE * scale)
        if size <= 0: return

        center_x = self.GRID_OFFSET_X + board_x * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = self.GRID_OFFSET_Y + board_y * self.TILE_SIZE + self.TILE_SIZE / 2

        rect = pygame.Rect(center_x - size/2, center_y - size/2, size, size)
        color = self.TILE_COLORS[tile_val]

        if alpha < 255:
            temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, (*color, alpha), (0, 0, *rect.size), border_radius=8)
            self.screen.blit(temp_surf, rect.topleft)
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=8)

    def _draw_selector(self):
        x, y = self.selector_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + x * self.TILE_SIZE,
            self.GRID_OFFSET_Y + y * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 3, border_radius=8)

    def _create_particles(self, board_x, board_y, tile_val):
        color = self.TILE_COLORS[tile_val]
        cx = self.GRID_OFFSET_X + board_x * self.TILE_SIZE + self.TILE_SIZE / 2
        cy = self.GRID_OFFSET_Y + board_y * self.TILE_SIZE + self.TILE_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append({
                'x': cx, 'y': cy, 'vx': vx, 'vy': vy,
                'color': color, 'life': self.np_random.integers(15, 30)
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            radius = int(max(0, p['life'] / 6))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, p['color'])
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 10))

        # Moves
        moves_text = self.font_main.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 10))

        # Game Over Message
        if self.game_over:
            is_win = self._check_win_condition()
            msg = "BOARD CLEARED!" if is_win else "OUT OF MOVES"
            color = (100, 255, 100) if is_win else (255, 100, 100)

            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_main.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

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
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Manual Play ---
    # Create a window to display the game
    pygame.display.set_caption("Match-3 Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement_action = 0
        space_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                if event.key == pygame.K_SPACE:
                    space_action = 1
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- GAME RESET ---")
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Only step if an action is taken
        if movement_action != 0 or space_action != 0:
            action = [movement_action, space_action, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            print(f"Info: {info}")

            if terminated:
                print("--- EPISODE FINISHED ---")
                # Wait for a moment before allowing reset
                pygame.time.wait(2000)

        # Update the display
        # The observation is already the rendered screen, so we just need to show it
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose it back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()