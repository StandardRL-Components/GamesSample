
# Generated: 2025-08-27T12:54:53.401653
# Source Brief: brief_00196.md
# Brief Index: 196

        
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
    """
    GameEnv is a puzzle game where the player pushes colored blocks onto matching
    targets within a grid. The objective is to solve the puzzle with a limited
    number of moves and within a time limit. The game emphasizes strategic
    planning, as pushing one block can cause a chain reaction, moving an entire
    line of blocks.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. "
        "Press space to push the block under the selector in the last direction you moved."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored blocks onto their matching targets. You have a limited number of moves and time. "
        "Plan your pushes carefully, as you'll move the entire row of blocks at once!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment, including game state, Pygame resources,
        and Gymnasium spaces.
        """
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 8
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        self.NUM_BLOCKS = 12
        self.MOVE_LIMIT = 20
        self.TIME_LIMIT_SECONDS = 60
        self.FPS = 30
        self.TIME_LIMIT_FRAMES = self.TIME_LIMIT_SECONDS * self.FPS
        self.ANIMATION_FRAMES = 8 # ~1/4 second animation

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
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 48, bold=True)

        # --- Colors ---
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_GRID = (50, 55, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.BLOCK_PALETTE = [
            (255, 99, 71), (255, 165, 0), (255, 215, 0), (50, 205, 50),
            (0, 191, 255), (30, 144, 255), (138, 43, 226), (218, 112, 214),
            (240, 128, 128), (72, 209, 204), (176, 224, 230), (255, 228, 181)
        ]

        # --- State Variables ---
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.moves_made = 0
        self.time_elapsed_frames = 0
        self.cursor_pos = [0, 0]
        self.last_move_direction = 1 # 1: up, 2: down, 3: left, 4: right
        self.blocks = []
        self.targets = []
        self.block_animations = {}
        self.space_was_held = False
        self.last_on_target_count = 0
        self.reward_since_last_action = 0
        self.win_message = ""

        # Initialize state
        self.reset()
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new episode.
        """
        super().reset(seed=seed)

        self.game_over = False
        self.steps = 0
        self.score = 0
        self.moves_made = 0
        self.time_elapsed_frames = 0
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_direction = 1
        self.block_animations = {}
        self.space_was_held = False
        self.reward_since_last_action = 0
        self.win_message = ""

        self._generate_puzzle()
        self.last_on_target_count = self._count_blocks_on_target()

        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        """
        Generates a new, solvable puzzle by creating a solved state and then
        applying a series of random, valid pushes.
        """
        self.blocks.clear()
        self.targets.clear()
        
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)
        
        chosen_colors = self.np_random.choice(len(self.BLOCK_PALETTE), self.NUM_BLOCKS, replace=False)

        initial_positions = all_coords[:self.NUM_BLOCKS]
        
        block_data = []
        for i in range(self.NUM_BLOCKS):
            pos = initial_positions[i]
            color_index = chosen_colors[i]
            color = self.BLOCK_PALETTE[color_index]
            target_color = tuple(c // 2 for c in color)

            self.targets.append({'pos': list(pos), 'color': target_color, 'id': i})
            block_data.append({'pos': list(pos), 'color': color, 'id': i})

        # Scramble the puzzle with random pushes
        scramble_moves = 15
        for _ in range(scramble_moves):
            if not block_data: continue
            
            block_to_push_idx = self.np_random.integers(0, len(block_data))
            push_direction = self.np_random.integers(1, 5) # 1-4 for directions
            
            block_pos = block_data[block_to_push_idx]['pos']
            
            self._apply_push_logic(block_data, block_pos, push_direction)

        self.blocks = block_data

    def _apply_push_logic(self, block_list, start_pos, direction):
        """
        Applies the push logic to a given list of blocks. This is used for both
        puzzle generation and gameplay.
        """
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
        
        line_of_blocks = []
        current_pos = list(start_pos)
        
        # Gather all blocks in the line of the push
        while any(b['pos'] == current_pos for b in block_list):
            block_in_line = next(b for b in block_list if b['pos'] == current_pos)
            line_of_blocks.append(block_in_line)
            current_pos[0] += dx
            current_pos[1] += dy
        
        if not line_of_blocks:
            return

        # Move all blocks in the line
        for block in line_of_blocks:
            block['pos'][0] = (block['pos'][0] + dx) % self.GRID_WIDTH
            block['pos'][1] = (block['pos'][1] + dy) % self.GRID_HEIGHT

    def step(self, action):
        """
        Advances the game state by one frame, processes the given action,
        and returns the standard Gymnasium 5-tuple.
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- 1. Update Time and Animations (Frame Tick) ---
        self.steps += 1
        self.time_elapsed_frames += 1
        animations_finished = self._update_animations()

        # If animations just finished, calculate reward for the completed move
        if animations_finished:
            reward += self._calculate_reward()

        # --- 2. Process Action ---
        movement, space_held, _ = action
        
        # Cursor movement
        if movement != 0 and not self.block_animations:
            self.last_move_direction = movement
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT
        
        # Push action on space press (rising edge)
        is_space_press = space_held and not self.space_was_held
        if is_space_press and not self.block_animations:
            block_to_push = self._get_block_at(self.cursor_pos)
            if block_to_push:
                self._handle_push(self.last_move_direction)
                self.moves_made += 1
                # sfx: block_push
                reward -= 0.1 # Small penalty for each move
            else:
                pass # sfx: push_fail
        
        self.space_was_held = bool(space_held)
        
        # --- 3. Check for Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.win_message == "VICTORY!":
                reward += 100 # Large reward for winning
            else:
                reward -= 10 # Penalty for losing

        self.score += reward
        
        # --- 4. Return API Tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_push(self, direction):
        """Initiates a push action for the block under the cursor."""
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
        
        line_of_blocks = []
        current_pos = list(self.cursor_pos)
        
        while True:
            block_in_line = self._get_block_at(current_pos)
            if not block_in_line:
                break
            line_of_blocks.append(block_in_line)
            current_pos[0] = (current_pos[0] + dx) % self.GRID_WIDTH
            current_pos[1] = (current_pos[1] + dy) % self.GRID_HEIGHT

        for block in line_of_blocks:
            start_pos = list(block['pos'])
            end_pos = [(start_pos[0] + dx) % self.GRID_WIDTH, (start_pos[1] + dy) % self.GRID_HEIGHT]
            self.block_animations[block['id']] = {
                'start': start_pos,
                'end': end_pos,
                'progress': 0,
            }

    def _update_animations(self):
        """Updates the state of all active animations. Returns True if any animation finished."""
        if not self.block_animations:
            return False
            
        finished_an_animation = False
        finished_ids = []
        for block_id, anim in self.block_animations.items():
            anim['progress'] += 1
            if anim['progress'] >= self.ANIMATION_FRAMES:
                block = next(b for b in self.blocks if b['id'] == block_id)
                block['pos'] = anim['end']
                finished_ids.append(block_id)
                finished_an_animation = True
        
        for block_id in finished_ids:
            del self.block_animations[block_id]
            
        return finished_an_animation

    def _calculate_reward(self):
        """Calculates reward based on the number of blocks on their targets."""
        on_target_count = self._count_blocks_on_target()
        reward = on_target_count - self.last_on_target_count
        self.last_on_target_count = on_target_count
        return float(reward)

    def _check_termination(self):
        """Checks for win or loss conditions."""
        if self.game_over:
            return True

        # Win condition
        if self._count_blocks_on_target() == self.NUM_BLOCKS:
            self.game_over = True
            self.win_message = "VICTORY!"
            # sfx: win_jingle
            return True

        # Loss conditions
        if self.moves_made >= self.MOVE_LIMIT:
            self.game_over = True
            self.win_message = "MOVES LIMIT"
            return True
        if self.time_elapsed_frames >= self.TIME_LIMIT_FRAMES:
            self.game_over = True
            self.win_message = "TIME'S UP"
            return True

        return False

    def _get_observation(self):
        """Renders the current game state to a Pygame surface and returns it as a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, targets, and blocks."""
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw targets
        for target in self.targets:
            rect = pygame.Rect(
                self.GRID_OFFSET_X + target['pos'][0] * self.CELL_SIZE + 4,
                self.GRID_OFFSET_Y + target['pos'][1] * self.CELL_SIZE + 4,
                self.CELL_SIZE - 8, self.CELL_SIZE - 8
            )
            pygame.draw.rect(self.screen, target['color'], rect, border_radius=4)
            
        # Draw blocks
        for block in self.blocks:
            px = [0,0]
            if block['id'] in self.block_animations:
                anim = self.block_animations[block['id']]
                prog = anim['progress'] / self.ANIMATION_FRAMES
                interp_prog = 1 - pow(1 - prog, 4)
                start_px = self._grid_to_pixel(anim['start'])
                end_px = self._grid_to_pixel(anim['end'])
                px[0] = start_px[0] + (end_px[0] - start_px[0]) * interp_prog
                px[1] = start_px[1] + (end_px[1] - start_px[1]) * interp_prog
            else:
                px = self._grid_to_pixel(block['pos'])
            self._draw_block((int(px[0]), int(px[1])), block['color'])

        # Draw cursor
        cursor_alpha = int(128 + 127 * math.sin(self.steps * 0.2))
        cursor_color = self.COLOR_CURSOR
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Create a temporary surface for transparency
        temp_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp_surface, (*cursor_color, cursor_alpha), temp_surface.get_rect(), 3, border_radius=5)
        self.screen.blit(temp_surface, cursor_rect.topleft)

    def _draw_block(self, pos_px, color):
        """Draws a single block with a 3D effect."""
        rect = pygame.Rect(pos_px[0] + 2, pos_px[1] + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        
        pygame.gfxdraw.box(self.screen, rect, color)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, shadow_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow_color, rect.topright, rect.bottomright, 2)

    def _render_ui(self):
        """Renders the UI text elements."""
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))

        moves_text = self.font_ui.render(f"MOVES: {self.moves_made}/{self.MOVE_LIMIT}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 15, 15))
        self.screen.blit(moves_text, moves_rect)
        
        time_left = max(0, self.TIME_LIMIT_SECONDS - self.time_elapsed_frames / self.FPS)
        time_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 15, 45))
        self.screen.blit(time_text, time_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg_color = (100, 255, 100) if self.win_message == "VICTORY!" else (255, 100, 100)
            msg_text = self.font_msg.render(self.win_message, True, msg_color)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        """Returns the info dictionary for the current state."""
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_made": self.moves_made,
            "time_elapsed_seconds": self.time_elapsed_frames / self.FPS,
            "blocks_on_target": self._count_blocks_on_target(),
        }

    def _grid_to_pixel(self, grid_pos):
        return (
            self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        )

    def _get_block_at(self, pos):
        for block in self.blocks:
            if block['pos'][0] == pos[0] and block['pos'][1] == pos[1]:
                return block
        return None

    def _count_blocks_on_target(self):
        count = 0
        for block in self.blocks:
            for target in self.targets:
                if block['id'] == target['id'] and block['pos'] == target['pos']:
                    count += 1
                    break
        return count

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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Block Pusher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    running = True
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
            
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    pygame.quit()