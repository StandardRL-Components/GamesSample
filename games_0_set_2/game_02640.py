
# Generated: 2025-08-27T21:00:00.955716
# Source Brief: brief_02640.md
# Brief Index: 2640

        
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
    """
    A match-3 puzzle game environment for Gymnasium.

    The player controls a cursor on an 8x8 grid of fruits. The goal is to
    swap adjacent fruits to form horizontal or vertical lines of three or
    more of the same type. Matched fruits are removed, and new fruits fall
    from the top, potentially creating chain reactions.

    The game is turn-based but features smooth animations for all actions.
    The state machine handles the sequence of swapping, matching, and falling
    fruits. An action from the agent can trigger this sequence, which then
    plays out over subsequent no-op steps until the board is idle again.

    The episode ends upon reaching a score of 1000 (win), running out of
    valid moves (loss), or reaching the maximum step limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a fruit. "
        "Move to an adjacent fruit and press space again to swap. Press shift to deselect."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent fruits to create lines of 3 or more. "
        "Clear fruits to score points and trigger cascades. The game ends if you reach 1000 points or run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_FRUITS = 5
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 1000

        # --- Visuals ---
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_HEIGHT * 40) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_WIDTH * 40) // 2 + 20
        self.CELL_SIZE = 40
        self.FRUIT_RADIUS = self.CELL_SIZE // 2 - 5
        self.ANIM_DURATION = 10 # frames

        # --- Colors ---
        self.COLOR_BG = (25, 35, 55)
        self.COLOR_GRID = (45, 55, 75)
        self.COLOR_CURSOR = (255, 255, 0, 100)
        self.COLOR_SELECT = (0, 255, 255, 120)
        self.COLOR_TEXT = (240, 240, 240)
        self.FRUIT_COLORS = [
            (220, 50, 50),   # Red
            (50, 200, 50),   # Green
            (60, 130, 255),  # Blue
            (255, 220, 50),  # Yellow
            (180, 60, 220),  # Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("sans-serif", 36, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 24)

        # --- State Variables ---
        self.board = None
        self.cursor_pos = None
        self.selected_pos = None
        self.last_action = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # --- Animation & State Machine ---
        self.game_state = 'IDLE' # 'IDLE', 'SWAP', 'MATCH', 'FALL'
        self.animations = []
        self._swapped_fruits_info = {}
        
        # This is a bit of a hack to give animation classes access to the env instance
        # without making them inner classes or passing `self` to every constructor.
        SwapAnimation.env = self
        MatchAnimation.env = self
        FallAnimation.env = self
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.last_action = np.array([0, 0, 0])
        self.game_state = 'IDLE'
        self.animations = []

        self._generate_board()

        return self._get_observation(), self._get_info()

    def _generate_board(self):
        self.board = self.np_random.integers(0, self.NUM_FRUITS, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            # Continously resolve matches until the board is stable
            while matches := self._find_all_matches():
                for r, c in matches:
                    self.board[r, c] = self.np_random.integers(0, self.NUM_FRUITS)
            
            # Check if the stable board has at least one valid move
            if self._find_valid_moves():
                break
            else:
                # If not, regenerate the whole board
                self.board = self.np_random.integers(0, self.NUM_FRUITS, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_state != 'IDLE':
            reward = self._update_animations_and_state()
        else:
            self._handle_input(action)

        self.last_action = action
        self.steps += 1

        if self.game_state == 'IDLE' and not self.game_over:
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                terminated = True
                reward += 100
                # Win sfx
            elif not self._find_valid_moves():
                self._reshuffle_board()
                if not self._find_valid_moves():
                    self.game_over = True
                    terminated = True
                    reward -= 50
                    # Lose sfx

        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        space_pressed = space_held == 1 and self.last_action[1] == 0
        shift_pressed = shift_held == 1 and self.last_action[2] == 0

        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        if shift_pressed and self.selected_pos:
            self.selected_pos = None # Deselect sfx

        if space_pressed:
            cx, cy = self.cursor_pos
            if self.selected_pos is None:
                self.selected_pos = [cx, cy] # Select sfx
            else:
                sx, sy = self.selected_pos
                if self._is_adjacent((cx, cy), (sx, sy)):
                    self.game_state = 'SWAP'
                    self._swapped_fruits_info = {'pos1': (sy, sx), 'pos2': (cy, cx)}
                    self._initiate_swap((sy, sx), (cy, cx))
                    # Swap sfx
                else:
                    self.selected_pos = [cx, cy] # Select sfx

    def _initiate_swap(self, pos1, pos2, swap_back=False):
        type1, type2 = self.board[pos1], self.board[pos2]
        self._swap_fruits(pos1, pos2)
        self.animations.append(SwapAnimation(pos1, pos2, type1, type2))
        if swap_back:
            self.game_state = 'SWAP_BACK'
        else:
            self.game_state = 'SWAP'

    def _update_animations_and_state(self):
        if self.animations:
            for anim in self.animations:
                anim.update()
            self.animations = [anim for anim in self.animations if not anim.is_finished()]
        
        if not self.animations:
            return self._on_animation_sequence_end()
        return 0

    def _on_animation_sequence_end(self):
        reward = 0
        if self.game_state == 'SWAP':
            matches = self._find_all_matches()
            if matches:
                reward += self._process_matches(matches, is_chain=False)
            else:
                pos1, pos2 = self._swapped_fruits_info['pos1'], self._swapped_fruits_info['pos2']
                self._initiate_swap(pos1, pos2, swap_back=True)
                reward = -0.2
        elif self.game_state == 'SWAP_BACK':
            self.game_state = 'IDLE'
            self.selected_pos = None
        elif self.game_state == 'MATCH':
            self._process_fall()
        elif self.game_state == 'FALL':
            matches = self._find_all_matches()
            if matches:
                reward += self._process_matches(matches, is_chain=True)
            else:
                self.game_state = 'IDLE'
                self.selected_pos = None
        return reward

    def _process_matches(self, matches, is_chain):
        # Match sfx
        reward = len(matches)
        if is_chain:
            reward += 5
        self.score += len(matches)
        
        for r, c in matches:
            self.animations.append(MatchAnimation((r, c)))
            self.board[r, c] = -1
        
        self.game_state = 'MATCH'
        return reward
            
    def _process_fall(self):
        self.game_state = 'FALL'
        for c in range(self.GRID_WIDTH):
            empty_count = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r, c] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    fruit_type = self.board[r, c]
                    self.board[r + empty_count, c] = fruit_type
                    self.board[r, c] = -1
                    self.animations.append(FallAnimation((r, c), (r + empty_count, c), fruit_type))
        
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.board[r, c] == -1:
                    fruit_type = self.np_random.integers(0, self.NUM_FRUITS)
                    self.board[r, c] = fruit_type
                    self.animations.append(FallAnimation((-1, c), (r, c), fruit_type))

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.board[r, c] != -1 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    for i in range(3): matches.add((r, c+i))
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.board[r, c] != -1 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    for i in range(3): matches.add((r+i, c))
        return list(matches)

    def _find_valid_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                        self._swap_fruits((r, c), (nr, nc))
                        if self._find_all_matches():
                            moves.append(((r, c), (nr, nc)))
                        self._swap_fruits((r, c), (nr, nc))
        return moves

    def _swap_fruits(self, pos1, pos2):
        self.board[pos1], self.board[pos2] = self.board[pos2], self.board[pos1]

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _reshuffle_board(self):
        flat_board = self.board.flatten()
        for _ in range(10):
            self.np_random.shuffle(flat_board)
            self.board = flat_board.reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_all_matches() and self._find_valid_moves():
                return
        # If still no valid board, game will end in main loop

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw static fruits
        animating_fruits = {pos for anim in self.animations for pos in anim.get_involved_positions()}
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in animating_fruits and self.board[r, c] != -1:
                    self._draw_fruit(c, r, self.board[r, c])
        
        # Draw animated elements
        for anim in self.animations:
            anim.draw()

        # Draw cursor
        cx, cy = self.cursor_pos
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), border_radius=5)
        self.screen.blit(s, (self.GRID_OFFSET_X + cx * self.CELL_SIZE, self.GRID_OFFSET_Y + cy * self.CELL_SIZE))

        # Draw selection
        if self.selected_pos:
            sx, sy = self.selected_pos
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_SELECT, s.get_rect(), border_radius=8, width=4)
            self.screen.blit(s, (self.GRID_OFFSET_X + sx * self.CELL_SIZE, self.GRID_OFFSET_Y + sy * self.CELL_SIZE))

    def _draw_fruit(self, c, r, fruit_type, offset_x=0, offset_y=0):
        if not (0 <= fruit_type < self.NUM_FRUITS): return
        center_x = int(self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE / 2 + offset_x)
        center_y = int(self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2 + offset_y)
        color = self.FRUIT_COLORS[fruit_type]
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.FRUIT_RADIUS, color)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.FRUIT_RADIUS, color)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            end_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "game_state": self.game_state}

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Animation:
    env = None
    def __init__(self, duration):
        self.duration = duration
        self.progress = 0
    def update(self): self.progress += 1
    def is_finished(self): return self.progress >= self.duration
    def get_involved_positions(self): return set()
    def draw(self): pass

class SwapAnimation(Animation):
    def __init__(self, pos1, pos2, type1, type2):
        super().__init__(self.env.ANIM_DURATION)
        self.r1, self.c1 = pos1
        self.r2, self.c2 = pos2
        self.type1, self.type2 = type1, type2
    def get_involved_positions(self): return {(self.r1, self.c1), (self.r2, self.c2)}
    def draw(self):
        p = self.progress / self.duration
        # Draw fruit 1 moving from pos1 to pos2
        ox1 = (self.c2 - self.c1) * p * self.env.CELL_SIZE
        oy1 = (self.r2 - self.r1) * p * self.env.CELL_SIZE
        self.env._draw_fruit(self.c1, self.r1, self.type1, ox1, oy1)
        # Draw fruit 2 moving from pos2 to pos1
        ox2 = (self.c1 - self.c2) * p * self.env.CELL_SIZE
        oy2 = (self.r1 - self.r2) * p * self.env.CELL_SIZE
        self.env._draw_fruit(self.c2, self.r2, self.type2, ox2, oy2)

class MatchAnimation(Animation):
    def __init__(self, pos):
        super().__init__(self.env.ANIM_DURATION)
        self.r, self.c = pos
    def draw(self):
        p = self.progress / self.duration
        radius = int(self.env.FRUIT_RADIUS * (1 - p))
        alpha = int(255 * (1 - p))
        center_x = int(self.env.GRID_OFFSET_X + self.c * self.env.CELL_SIZE + self.env.CELL_SIZE / 2)
        center_y = int(self.env.GRID_OFFSET_Y + self.r * self.env.CELL_SIZE + self.env.CELL_SIZE / 2)
        pygame.gfxdraw.aacircle(self.env.screen, center_x, center_y, radius, (255, 255, 255, alpha))
        pygame.gfxdraw.filled_circle(self.env.screen, center_x, center_y, radius, (255, 255, 255, alpha))

class FallAnimation(Animation):
    def __init__(self, start_pos, end_pos, fruit_type):
        super().__init__(self.env.ANIM_DURATION)
        self.start_r, self.start_c = start_pos
        self.end_r, self.end_c = end_pos
        self.fruit_type = fruit_type
    def get_involved_positions(self): return {(self.end_r, self.end_c)}
    def draw(self):
        p = min(1.0, self.progress / self.duration)
        current_r = self.start_r + (self.end_r - self.start_r) * p
        self.env._draw_fruit(self.end_c, current_r, self.fruit_type)

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Matcher")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action = np.array([0, 0, 0])
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r: obs, info = env.reset()

        if env.game_state == 'IDLE':
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
            
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, State: {info['game_state']}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()