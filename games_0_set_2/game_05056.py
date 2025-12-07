
# Generated: 2025-08-28T03:50:15.278335
# Source Brief: brief_05056.md
# Brief Index: 5056

        
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
        "Controls: Arrow keys to move cursor. Space to select a block and match it. Shift to reshuffle the board (costs 1 move)."
    )

    game_description = (
        "Match 3 or more adjacent colored blocks to score points. Create chain reactions for big bonuses! Reach 1000 points in 20 moves to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 8
    BLOCK_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * BLOCK_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * BLOCK_SIZE) // 2
    NUM_COLORS = 5
    TARGET_SCORE = 1000
    STARTING_MOVES = 20
    MIN_MATCH_SIZE = 3

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)
    COLORS = [
        (0, 0, 0),  # 0: Empty
        (231, 76, 60),   # 1: Red
        (46, 204, 113),  # 2: Green
        (52, 152, 219),  # 3: Blue
        (241, 196, 15),  # 4: Yellow
        (155, 89, 182),  # 5: Purple
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_popup = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_gameover = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = None
        self.floating_texts = None
        self.rng = None
        self.steps = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.score = 0
        self.moves_remaining = self.STARTING_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.floating_texts = []
        self.steps = 0

        # Generate a valid starting board
        while True:
            self._fill_board()
            if self._has_valid_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        move_made = False

        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # --- Handle Cursor Movement ---
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

            # --- Handle Block Selection ---
            if space_held and not self.prev_space_held:
                turn_reward, blocks_cleared = self._process_selection()
                if blocks_cleared > 0:
                    reward += turn_reward
                    self.score += turn_reward
                    move_made = True
                # SFX: play 'select_fail' or 'select_success' sound

            # --- Handle Reshuffle ---
            if shift_held and not self.prev_shift_held:
                if self.moves_remaining > 0:
                    self._reshuffle_board()
                    move_made = True
                    # SFX: play 'reshuffle' sound

            if move_made:
                self.moves_remaining -= 1
                if not self._has_valid_moves() and self.moves_remaining > 0:
                    self._reshuffle_board() # Free reshuffle if stuck
                    # SFX: play 'auto_reshuffle' sound
            
            self.prev_space_held = space_held
            self.prev_shift_held = shift_held

        self._update_effects()
        terminated = self._check_termination()

        if terminated and not self.game_over: # Terminal reward on first frame of termination
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Win bonus
                # SFX: play 'win_game' sound
            else:
                reward -= 50 # Loss penalty
                # SFX: play 'lose_game' sound
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_selection(self):
        start_x, start_y = self.cursor_pos
        start_color = self.grid[start_y][start_x]
        if start_color == 0: return 0, 0

        connected = self._find_connected_blocks(start_x, start_y, start_color)
        if len(connected) < self.MIN_MATCH_SIZE:
            return 0, 0

        turn_reward = 0
        total_blocks_cleared = 0
        chain_level = 1

        # Initial match
        cleared_this_chain = self._clear_blocks(connected)
        turn_reward += cleared_this_chain
        if cleared_this_chain > 3: turn_reward += 10
        total_blocks_cleared += cleared_this_chain
        # SFX: play 'match_chain_1' sound

        # Process chain reactions
        while True:
            self._apply_gravity()
            self._fill_top_rows()
            
            chain_matches = self._find_all_matches()
            if not chain_matches:
                break
            
            chain_level += 1
            # SFX: play f'match_chain_{min(chain_level, 5)}' sound
            
            cleared_this_chain = self._clear_blocks(chain_matches)
            turn_reward += cleared_this_chain * chain_level # Combo bonus
            if cleared_this_chain > 3: turn_reward += 10 * chain_level
            total_blocks_cleared += cleared_this_chain
        
        return turn_reward, total_blocks_cleared

    def _clear_blocks(self, blocks_to_clear):
        for x, y in blocks_to_clear:
            color_index = self.grid[y][x]
            if color_index > 0:
                self._create_particles(x, y, color_index)
                self.grid[y][x] = 0
        
        if blocks_to_clear:
            # Create a single floating text for the whole group
            avg_x = sum(p[0] for p in blocks_to_clear) / len(blocks_to_clear)
            avg_y = sum(p[1] for p in blocks_to_clear) / len(blocks_to_clear)
            score_val = len(blocks_to_clear)
            if len(blocks_to_clear) > 3: score_val += 10
            self._create_floating_text(f"+{score_val}", avg_x, avg_y)

        return len(blocks_to_clear)

    def _find_connected_blocks(self, start_x, start_y, color):
        if color == 0: return []
        q = [(start_x, start_y)]
        visited = set(q)
        connected = []
        while q:
            x, y = q.pop(0)
            connected.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[ny][nx] == color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return connected

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] != 0:
                    if y != empty_row:
                        self.grid[empty_row][x] = self.grid[y][x]
                        self.grid[y][x] = 0
                    empty_row -= 1
    
    def _fill_top_rows(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == 0:
                    self.grid[y][x] = self.rng.integers(1, self.NUM_COLORS + 1)

    def _fill_board(self):
        self.grid = self.rng.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH)).tolist()

    def _reshuffle_board(self):
        blocks = [self.grid[y][x] for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH)]
        self.rng.shuffle(blocks)
        
        while True:
            self.grid = [blocks[i*self.GRID_WIDTH:(i+1)*self.GRID_WIDTH] for i in range(self.GRID_HEIGHT)]
            if self._has_valid_moves():
                break
            self.rng.shuffle(blocks)

    def _find_all_matches(self):
        all_matches = set()
        visited = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in visited:
                    color = self.grid[y][x]
                    if color > 0:
                        connected = self._find_connected_blocks(x, y, color)
                        for block in connected:
                            visited.add(block)
                        if len(connected) >= self.MIN_MATCH_SIZE:
                            for block in connected:
                                all_matches.add(block)
        return list(all_matches)

    def _has_valid_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.grid[y][x]
                if color > 0:
                    if len(self._find_connected_blocks(x, y, color)) >= self.MIN_MATCH_SIZE:
                        return True
        return False

    def _check_termination(self):
        return self.score >= self.TARGET_SCORE or self.moves_remaining <= 0
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Draw blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_index = self.grid[y][x]
                if color_index > 0:
                    color = self.COLORS[color_index]
                    darker_color = tuple(max(0, c - 40) for c in color)
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + x * self.BLOCK_SIZE,
                        self.GRID_OFFSET_Y + y * self.BLOCK_SIZE,
                        self.BLOCK_SIZE, self.BLOCK_SIZE
                    )
                    pygame.gfxdraw.box(self.screen, rect.inflate(-4, -4), color)
                    pygame.gfxdraw.rectangle(self.screen, rect.inflate(-4, -4), darker_color)

        # Draw cursor
        cursor_flash = self.steps % 30 < 20
        if cursor_flash:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(
                self.GRID_OFFSET_X + cx * self.BLOCK_SIZE,
                self.GRID_OFFSET_Y + cy * self.BLOCK_SIZE,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw particles and floating texts
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))
        for ft in self.floating_texts:
            text_surf = self.font_popup.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(ft['alpha'])
            self.screen.blit(text_surf, (int(ft['pos'][0] - text_surf.get_width() / 2), int(ft['pos'][1])))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves
        moves_text = self.font_main.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 10))

        # Game Over / Win overlay
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.TARGET_SCORE:
                msg = "YOU WIN!"
                color = self.COLORS[2] # Green
            else:
                msg = "GAME OVER"
                color = self.COLORS[1] # Red

            end_text = self.font_gameover.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _create_particles(self, grid_x, grid_y, color_index):
        px = self.GRID_OFFSET_X + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        color = self.COLORS[color_index]
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'radius': random.uniform(2, 5),
                'lifetime': random.randint(20, 40),
                'color': color
            })

    def _create_floating_text(self, text, grid_x, grid_y):
        px = self.GRID_OFFSET_X + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        self.floating_texts.append({
            'text': text,
            'pos': [px, py],
            'lifetime': 60,
            'alpha': 255,
            'color': (255, 255, 100)
        })

    def _update_effects(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['radius'] *= 0.95
            if p['lifetime'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)
        
        # Update floating texts
        for ft in self.floating_texts[:]:
            ft['pos'][1] -= 0.5
            ft['lifetime'] -= 1
            ft['alpha'] = max(0, int(255 * (ft['lifetime'] / 60)))
            if ft['lifetime'] <= 0:
                self.floating_texts.remove(ft)

    def close(self):
        pygame.quit()

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Block Matcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop for human control
    while not done:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Reward: {reward}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # We need a small delay for human play, otherwise it runs too fast
        clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()