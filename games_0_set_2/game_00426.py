# Generated: 2025-08-27T13:36:55.147591
# Source Brief: brief_00426.md
# Brief Index: 426

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrows to move the cursor. Press Space to select a gem. "
        "Move the cursor to an adjacent gem and press Space again to swap. "
        "Hold Shift to reshuffle the board (costs 10 moves)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap gems to create lines of 3 or more. "
        "Reach the target score before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    BOARD_SIZE = 8
    NUM_GEM_TYPES = 6
    GEM_SIZE = 40
    BOARD_OFFSET_X = (SCREEN_WIDTH - BOARD_SIZE * GEM_SIZE) // 2
    BOARD_OFFSET_Y = (SCREEN_HEIGHT - BOARD_SIZE * GEM_SIZE) // 2
    WIN_SCORE = 1000
    STARTING_MOVES = 50
    RESHUFFLE_COST = 10
    
    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTION = (255, 255, 255)
    GEM_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 150, 255),  # Blue
        (255, 255, 50),  # Yellow
        (255, 50, 255),  # Magenta
        (50, 255, 255),  # Cyan
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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.board = None
        self.cursor_pos = None
        self.selected_gem = None
        self.last_move_dir = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.game_state = "IDLE" # IDLE, SWAPPING, MATCHING, DROPPING
        self.animations = []
        self.particles = []
        self.reward_buffer = 0
        self.gems_to_clear = set()
        
        # The reset method is called here, which will properly initialize the RNG
        # and other game state variables.
        # self.reset() is not called here to avoid double-initialization issues
        # with some environment wrappers. It's standard practice to let the
        # user call reset() explicitly after __init__().

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.STARTING_MOVES
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.BOARD_SIZE // 2, self.BOARD_SIZE // 2]
        self.selected_gem = None
        self.last_move_dir = 1  # Default to 'up'
        
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True

        self.game_state = "IDLE"
        self.animations = []
        self.particles = []
        self.reward_buffer = 0
        self.gems_to_clear = set()
        
        self._generate_board()
        while not self._find_possible_moves():
            self._generate_board()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.reward_buffer = 0

        if self.game_state != "IDLE":
            self._update_game_state()
        else:
            if self.game_over:
                # Do nothing if game is over and waiting
                pass
            else:
                self._handle_action(action)
                self._update_game_state() # Start animations if any were created

        terminated = self.game_over
        
        # Check for win/loss conditions if not already over
        if not self.game_over:
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                self.win = True
                self.reward_buffer += 100
                # sfx: game_win
            elif self.moves_left <= 0:
                self.game_over = True
                self.win = False
                self.reward_buffer -= 10
                # sfx: game_lose

        # Anti-softlock: if idle and no moves are possible, reshuffle
        if self.game_state == "IDLE" and not self.game_over and not self._find_possible_moves():
            self._reshuffle_board(free=True)

        return (
            self._get_observation(),
            self.reward_buffer,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Movement ---
        if movement > 0:
            self.last_move_dir = movement
            if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.BOARD_SIZE) % self.BOARD_SIZE
            elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.BOARD_SIZE
            elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.BOARD_SIZE) % self.BOARD_SIZE
            elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.BOARD_SIZE
            # sfx: cursor_move

        # --- Shift Action ---
        if shift_press and self.moves_left >= self.RESHUFFLE_COST:
            self.moves_left -= self.RESHUFFLE_COST
            self._reshuffle_board()
            self.reward_buffer -= 1.0 # Costly action
            # sfx: board_reshuffle
            return

        # --- Space Action ---
        if space_press:
            if self.selected_gem is None:
                # Select a gem
                self.selected_gem = list(self.cursor_pos)
                # sfx: gem_select
            else:
                # Attempt to swap
                target_pos = self._get_target_from_dir(self.selected_gem, self.last_move_dir)
                if target_pos == self.cursor_pos:
                    self._initiate_swap(self.selected_gem, target_pos)
                    self.selected_gem = None
                else:
                    # Invalid selection, just change selection to new cursor pos
                    self.selected_gem = list(self.cursor_pos)
                    # sfx: invalid_selection

    def _update_game_state(self):
        if not self.animations:
            if self.game_state == "SWAPPING":
                # Swap animation finished, now check for matches
                matches = self._find_all_matches()
                if self.swap_data['is_valid_swap']:
                    if matches:
                        self._process_matches(matches)
                        # sfx: match_found
                    else:
                        # Invalid swap, swap back
                        self.reward_buffer -= 0.1
                        self._initiate_swap(self.swap_data['pos2'], self.swap_data['pos1'], is_valid=False)
                        # sfx: invalid_swap
                else:
                    # Invalid swap finished returning, go idle
                    self.game_state = "IDLE"
            elif self.game_state == "MATCHING":
                # Gems finished flashing, now clear them
                self._clear_gems()
                self.game_state = "DROPPING"
            elif self.game_state == "DROPPING":
                # Gems finished dropping, check for new chain reactions
                matches = self._find_all_matches()
                if matches:
                    self._process_matches(matches)
                    # sfx: chain_reaction
                else:
                    self.game_state = "IDLE"
        
        self._update_animations()

    def _initiate_swap(self, pos1, pos2, is_valid=True):
        if not self._is_adjacent(pos1, pos2):
            return
            
        if is_valid:
            self.moves_left -= 1

        self.game_state = "SWAPPING"
        self.swap_data = {'pos1': pos1, 'pos2': pos2, 'is_valid_swap': is_valid}

        gem1_type = self.board[pos1[1], pos1[0]]
        gem2_type = self.board[pos2[1], pos2[0]]
        
        self.board[pos1[1], pos1[0]] = gem2_type
        self.board[pos2[1], pos2[0]] = gem1_type

        self.animations.append({
            'type': 'move', 'gem_type': gem2_type,
            'start_pos': pos2, 'end_pos': pos1, 'progress': 0.0
        })
        self.animations.append({
            'type': 'move', 'gem_type': gem1_type,
            'start_pos': pos1, 'end_pos': pos2, 'progress': 0.0
        })
        # sfx: gem_swap

    def _process_matches(self, matches):
        self.game_state = "MATCHING"
        self.gems_to_clear = set()
        total_matched_gems = 0
        
        for match in matches:
            total_matched_gems += len(match)
            if len(match) == 4: self.reward_buffer += 2
            elif len(match) >= 5: self.reward_buffer += 3
            else: self.reward_buffer += 1

            for pos in match:
                self.gems_to_clear.add(pos)
                self.animations.append({'type': 'flash', 'pos': pos, 'progress': 0.0})

        self.score += total_matched_gems * 10
        if len(matches) > 1: # Combo bonus
            self.score += len(matches) * 25
            self.reward_buffer += len(matches) # Bonus for combos

    def _clear_gems(self):
        for x, y in self.gems_to_clear:
            gem_type = self.board[y, x]
            if gem_type != -1:
                self._spawn_particles(x, y, gem_type)
                self.board[y, x] = -1 # -1 signifies empty
        # sfx: gem_clear

        # Now, make gems fall
        for x in range(self.BOARD_SIZE):
            empty_count = 0
            for y in range(self.BOARD_SIZE - 1, -1, -1):
                if self.board[y, x] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    gem_type = self.board[y, x]
                    self.board[y + empty_count, x] = gem_type
                    self.board[y, x] = -1
                    self.animations.append({
                        'type': 'move', 'gem_type': gem_type,
                        'start_pos': [x, y], 'end_pos': [x, y + empty_count], 'progress': 0.0
                    })
        
        # Fill new gems from top
        for x in range(self.BOARD_SIZE):
            for y in range(self.BOARD_SIZE):
                if self.board[y, x] == -1:
                    gem_type = self.np_random.integers(0, self.NUM_GEM_TYPES)
                    self.board[y, x] = gem_type
                    start_y = y - self.BOARD_SIZE
                    self.animations.append({
                        'type': 'move', 'gem_type': gem_type,
                        'start_pos': [x, start_y], 'end_pos': [x, y], 'progress': 0.0
                    })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        
        # Draw static gems
        animating_gems_pos = [tuple(anim['end_pos']) for anim in self.animations if anim['type'] == 'move']
        flashing_gems_pos = [tuple(anim['pos']) for anim in self.animations if anim['type'] == 'flash']
        
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                pos = (x, y)
                if pos not in animating_gems_pos and self.board[y, x] != -1:
                    gem_type = self.board[y, x]
                    alpha = 255
                    size_mult = 1.0
                    if pos in flashing_gems_pos:
                        for anim in self.animations:
                            if anim['type'] == 'flash' and tuple(anim['pos']) == pos:
                                size_mult = 1.0 + 0.3 * math.sin(anim['progress'] * math.pi)
                                break
                    self._draw_gem(x, y, gem_type, size_mult, alpha)
        
        # Draw animating gems
        for anim in self.animations:
            if anim['type'] == 'move':
                prog = anim['progress']
                start_x, start_y = anim['start_pos']
                end_x, end_y = anim['end_pos']
                draw_x = start_x + (end_x - start_x) * prog
                draw_y = start_y + (end_y - start_y) * prog
                self._draw_gem(draw_x, draw_y, anim['gem_type'], 1.0, 255)

        self._update_and_draw_particles()
        self._draw_cursor_and_selection()

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Moves
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 10))
        
        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Helper & Drawing Methods ---
    
    def _generate_board(self):
        self.board = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.BOARD_SIZE, self.BOARD_SIZE))
        # Prevent initial matches
        while self._find_all_matches():
            matches = self._find_all_matches()
            for match in matches:
                for x, y in match:
                    # Replace one gem in the match to break it
                    self.board[y, x] = (self.board[y, x] + 1) % self.NUM_GEM_TYPES
                    break

    def _draw_grid(self):
        for i in range(self.BOARD_SIZE + 1):
            # Vertical
            start_pos = (self.BOARD_OFFSET_X + i * self.GEM_SIZE, self.BOARD_OFFSET_Y)
            end_pos = (self.BOARD_OFFSET_X + i * self.GEM_SIZE, self.BOARD_OFFSET_Y + self.BOARD_SIZE * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.BOARD_OFFSET_X, self.BOARD_OFFSET_Y + i * self.GEM_SIZE)
            end_pos = (self.BOARD_OFFSET_X + self.BOARD_SIZE * self.GEM_SIZE, self.BOARD_OFFSET_Y + i * self.GEM_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_gem(self, x, y, gem_type, size_mult, alpha):
        if gem_type < 0 or gem_type >= self.NUM_GEM_TYPES: return
        
        px = self.BOARD_OFFSET_X + x * self.GEM_SIZE + self.GEM_SIZE / 2
        py = self.BOARD_OFFSET_Y + y * self.GEM_SIZE + self.GEM_SIZE / 2
        radius = int(self.GEM_SIZE * 0.4 * size_mult)
        color = self.GEM_COLORS[gem_type]
        
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, color)
        
        # Shine effect
        shine_radius = int(radius * 0.4)
        shine_px = int(px - radius * 0.3)
        shine_py = int(py - radius * 0.3)
        shine_color = (255, 255, 255, 150)
        pygame.gfxdraw.filled_circle(self.screen, shine_px, shine_py, shine_radius, shine_color)

    def _draw_cursor_and_selection(self):
        # Draw selection
        if self.selected_gem:
            x, y = self.selected_gem
            rect = pygame.Rect(self.BOARD_OFFSET_X + x * self.GEM_SIZE,
                               self.BOARD_OFFSET_Y + y * self.GEM_SIZE,
                               self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 3)

        # Draw cursor
        x, y = self.cursor_pos
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        thickness = int(2 + pulse * 2)
        rect = pygame.Rect(self.BOARD_OFFSET_X + x * self.GEM_SIZE,
                           self.BOARD_OFFSET_Y + y * self.GEM_SIZE,
                           self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, thickness)

    def _update_animations(self):
        dt = 0.1 # Animation speed
        
        # Use a while loop as we might remove items
        i = 0
        while i < len(self.animations):
            anim = self.animations[i]
            anim['progress'] += dt
            if anim['progress'] >= 1.0:
                self.animations.pop(i)
            else:
                i += 1

    def _spawn_particles(self, x, y, gem_type):
        px = self.BOARD_OFFSET_X + x * self.GEM_SIZE + self.GEM_SIZE / 2
        py = self.BOARD_OFFSET_Y + y * self.GEM_SIZE + self.GEM_SIZE / 2
        color = self.GEM_COLORS[gem_type]

        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': 1.0, 'color': color})

    def _update_and_draw_particles(self):
        i = 0
        while i < len(self.particles):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.04
            
            if p['life'] <= 0:
                self.particles.pop(i)
            else:
                radius = int(p['life'] * 5)
                color = (*p['color'], int(p['life'] * 255))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)
                i += 1

    def _find_all_matches(self):
        matches = []
        matched_gems = set()
        
        # Horizontal
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE - 2):
                if (x,y) not in matched_gems and self.board[y,x] != -1:
                    if self.board[y,x] == self.board[y,x+1] == self.board[y,x+2]:
                        match = [(x,y), (x+1,y), (x+2,y)]
                        # Extend match
                        for i in range(x + 3, self.BOARD_SIZE):
                            if self.board[y,x] == self.board[y,i]:
                                match.append((i,y))
                            else:
                                break
                        matches.append(match)
                        for pos in match: matched_gems.add(pos)
        
        # Vertical
        for x in range(self.BOARD_SIZE):
            for y in range(self.BOARD_SIZE - 2):
                if (x,y) not in matched_gems and self.board[y,x] != -1:
                    if self.board[y,x] == self.board[y+1,x] == self.board[y+2,x]:
                        match = [(x,y), (x,y+1), (x,y+2)]
                        for i in range(y + 3, self.BOARD_SIZE):
                            if self.board[y,x] == self.board[i,x]:
                                match.append((x,i))
                            else:
                                break
                        matches.append(match)
                        for pos in match: matched_gems.add(pos)
        return matches

    def _find_possible_moves(self):
        moves = []
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                # Try swapping right
                if x < self.BOARD_SIZE - 1:
                    self.board[y,x], self.board[y,x+1] = self.board[y,x+1], self.board[y,x]
                    if self._find_all_matches(): moves.append(((x,y), (x+1,y)))
                    self.board[y,x], self.board[y,x+1] = self.board[y,x+1], self.board[y,x] # Swap back
                # Try swapping down
                if y < self.BOARD_SIZE - 1:
                    self.board[y,x], self.board[y+1,x] = self.board[y+1,x], self.board[y,x]
                    if self._find_all_matches(): moves.append(((x,y), (x,y+1)))
                    self.board[y,x], self.board[y+1,x] = self.board[y+1,x], self.board[y,x] # Swap back
        return moves

    def _reshuffle_board(self, free=False):
        flat_board = self.board.flatten().tolist()
        self.np_random.shuffle(flat_board)
        self.board = np.array(flat_board).reshape((self.BOARD_SIZE, self.BOARD_SIZE))
        
        while self._find_all_matches() or not self._find_possible_moves():
            self.np_random.shuffle(flat_board)
            self.board = np.array(flat_board).reshape((self.BOARD_SIZE, self.BOARD_SIZE))

        # Add a visual effect for the reshuffle
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                self.animations.append({'type': 'flash', 'pos': (x,y), 'progress': 0.0})
        self.game_state = "MATCHING" # Use this state to handle the flash animation

    def _get_target_from_dir(self, pos, direction):
        x, y = pos
        if direction == 1: y = (y - 1 + self.BOARD_SIZE) % self.BOARD_SIZE
        elif direction == 2: y = (y + 1) % self.BOARD_SIZE
        elif direction == 3: x = (x - 1 + self.BOARD_SIZE) % self.BOARD_SIZE
        elif direction == 4: x = (x + 1) % self.BOARD_SIZE
        return [x, y]

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # The main block is for human play and requires a display.
    # It will not run in the headless testing environment.
    # To run this, you would need to unset the SDL_VIDEODRIVER variable.
    # Example:
    # import os
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    import time
    
    # For human play
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up Pygame window for display
    pygame.display.set_caption("Gem Swap")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    
    # Game loop
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Handle Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        # --- Step the environment ---
        # Since auto_advance is False, we need to decide when to step.
        # For a turn-based game, we step on any key press.
        # To make it feel responsive, we'll step every frame.
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
        
        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        time.sleep(1/30) # Limit to 30 FPS

    env.close()