
# Generated: 2025-08-27T20:44:06.814355
# Source Brief: brief_02558.md
# Brief Index: 2558

        
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
    An isometric puzzle game where the player pushes colored boxes onto matching targets.
    The goal is to solve the puzzle with the minimum number of moves to maximize the score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push the selected box. Press Space to cycle which box is selected."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored boxes onto matching targets in this isometric puzzle game. Plan your moves carefully to maximize your score before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_BG = (35, 45, 55)
        self.COLOR_GRID = (60, 70, 80)
        self.COLOR_TEXT = (230, 230, 240)
        self.BOX_COLORS = [
            (227, 87, 78),    # Red
            (87, 199, 110),   # Green
            (78, 133, 227),   # Blue
            (230, 217, 109),  # Yellow
            (109, 230, 223),  # Cyan
            (227, 78, 201),   # Magenta
        ]

        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 8
        self.NUM_BOXES = 6
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000

        # Isometric projection constants
        self.TILE_WIDTH_ISO = 64
        self.TILE_HEIGHT_ISO = 32
        self.BOX_HEIGHT_PX = 32
        self.ORIGIN_X = 640 // 2
        self.ORIGIN_Y = 60
        
        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.box_positions = []
        self.target_positions = []
        self.box_on_target = []
        self.selected_box_index = 0
        self.prev_space_held = False
        self.animation_tick = 0
        
        # Pre-defined levels
        self._levels = self._create_levels()
        
        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def _create_levels(self):
        level1 = {
            "targets": [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)],
            "boxes":   [(2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]
        }
        level2 = {
            "targets": [(1, 1), (1, 2), (1, 3), (8, 4), (8, 5), (8, 6)],
            "boxes":   [(4, 4), (4, 3), (4, 2), (5, 5), (5, 6), (5, 7)]
        }
        level3 = {
            "targets": [(1, 6), (2, 6), (3, 6), (6, 1), (7, 1), (8, 1)],
            "boxes":   [(1, 1), (2, 1), (3, 1), (6, 6), (7, 6), (8, 6)]
        }
        return [level1, level2, level3]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.selected_box_index = 0
        self.prev_space_held = True # Prevent action on first frame after reset

        # Pick a random level
        level_idx = self.np_random.integers(0, len(self._levels))
        level = self._levels[level_idx]
        
        self.box_positions = [pos for pos in level["boxes"]]
        self.target_positions = [pos for pos in level["targets"]]
        
        self.box_on_target = [self._is_box_on_target(i) for i in range(self.NUM_BOXES)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused per brief
        
        reward = 0
        
        # 1. Handle selection change (on space press, not hold)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            self.selected_box_index = (self.selected_box_index + 1) % self.NUM_BOXES
            # sfx: select_cycle.wav
        self.prev_space_held = space_held

        # 2. Handle box movement
        if movement != 0:
            box_idx = self.selected_box_index
            old_pos = self.box_positions[box_idx]
            
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)

            # Check for valid move
            is_valid_move = True
            if not (0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT):
                is_valid_move = False
            if new_pos in self.box_positions:
                is_valid_move = False

            if is_valid_move:
                # sfx: box_push.wav
                reward += self._calculate_move_reward(box_idx, new_pos)
                
                self.box_positions[box_idx] = new_pos
                self.moves_remaining -= 1
                
                was_on_target = self.box_on_target[box_idx]
                is_on_target = self._is_box_on_target(box_idx)
                if is_on_target and not was_on_target:
                    reward += 10
                    # sfx: box_lock.wav
                if not is_on_target and was_on_target:
                    reward -= 10
                self.box_on_target[box_idx] = is_on_target
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and self._check_win_condition():
            win_bonus = 50
            reward += win_bonus
            self.score += win_bonus
            # sfx: puzzle_solved.wav
        elif terminated and not self._check_win_condition():
            # sfx: puzzle_failed.wav
            pass

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_move_reward(self, box_idx, new_pos):
        old_pos = self.box_positions[box_idx]
        target_pos = self.target_positions[box_idx]
        
        old_dist = abs(old_pos[0] - target_pos[0]) + abs(old_pos[1] - target_pos[1])
        new_dist = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])

        if new_dist < old_dist: return 1
        if new_dist > old_dist: return -1
        return 0

    def _is_box_on_target(self, box_idx):
        return self.box_positions[box_idx] == self.target_positions[box_idx]

    def _check_win_condition(self):
        return all(self.box_on_target)

    def _check_termination(self):
        if self.game_over: return True
        
        win = self._check_win_condition()
        loss_moves = self.moves_remaining <= 0
        loss_steps = self.steps >= self.MAX_STEPS

        if win or loss_moves or loss_steps:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        self.animation_tick += 1
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
            "puzzle_solved": self._check_win_condition(),
        }

    def _grid_to_iso(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _render_game(self):
        self._render_grid()
        
        render_list = []
        for i in range(self.NUM_BOXES):
            render_list.append(('T', i, self.target_positions[i]))
            render_list.append(('B', i, self.box_positions[i]))
        
        render_list.sort(key=lambda item: item[2][0] + item[2][1] + (0.1 if item[1]=='B' else 0))

        for item_type, idx, pos in render_list:
            if item_type == 'T':
                self._render_target(idx, pos)
            elif item_type == 'B':
                self._render_box(idx, pos)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_iso(0, y)
            end = self._grid_to_iso(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._grid_to_iso(x, 0)
            end = self._grid_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_target(self, idx, pos):
        color = self.BOX_COLORS[idx]
        center_x, center_y = self._grid_to_iso(pos[0], pos[1])
        
        points = [
            (center_x, center_y - self.TILE_HEIGHT_ISO // 2),
            (center_x + self.TILE_WIDTH_ISO // 2, center_y),
            (center_x, center_y + self.TILE_HEIGHT_ISO // 2),
            (center_x - self.TILE_WIDTH_ISO // 2, center_y),
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, (*color, 100))
        pygame.gfxdraw.aapolygon(self.screen, points, (*color, 150))

    def _render_box(self, idx, pos):
        color = self.BOX_COLORS[idx]
        color_light = tuple(min(255, c + 30) for c in color)
        color_dark = tuple(max(0, c - 40) for c in color)
        
        center_x, center_y = self._grid_to_iso(pos[0], pos[1])
        center_y -= self.BOX_HEIGHT_PX // 2
        
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        
        top_face = [
            (center_x, center_y - h // 2),
            (center_x + w // 2, center_y),
            (center_x, center_y + h // 2),
            (center_x - w // 2, center_y),
        ]
        left_face = [
            (center_x - w // 2, center_y),
            (center_x, center_y + h // 2),
            (center_x, center_y + h // 2 + self.BOX_HEIGHT_PX),
            (center_x - w // 2, center_y + self.BOX_HEIGHT_PX),
        ]
        right_face = [
            (center_x + w // 2, center_y),
            (center_x, center_y + h // 2),
            (center_x, center_y + h // 2 + self.BOX_HEIGHT_PX),
            (center_x + w // 2, center_y + self.BOX_HEIGHT_PX),
        ]
        
        # Render selector pulse if this box is selected
        if idx == self.selected_box_index:
            pulse = (math.sin(self.animation_tick * 0.1) + 1) / 2
            radius = int(self.TILE_WIDTH_ISO * 0.5 * (0.9 + pulse * 0.15))
            alpha = int(100 + pulse * 50)
            sel_color = (255, 255, 255)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y + h // 2 + self.BOX_HEIGHT_PX, radius, (*sel_color, alpha // 4))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y + h // 2 + self.BOX_HEIGHT_PX, radius, (*sel_color, alpha))

        pygame.gfxdraw.filled_polygon(self.screen, left_face, color_dark)
        pygame.gfxdraw.filled_polygon(self.screen, right_face, color)
        pygame.gfxdraw.filled_polygon(self.screen, top_face, color_light)
        
        pygame.gfxdraw.aapolygon(self.screen, left_face, tuple(max(0, c-20) for c in color_dark))
        pygame.gfxdraw.aapolygon(self.screen, right_face, tuple(max(0, c-20) for c in color))
        pygame.gfxdraw.aapolygon(self.screen, top_face, tuple(max(0, c-20) for c in color_light))

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {self.moves_remaining}"
        text_surf = self.font_large.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 15))
        
        # Score
        score_text = f"Score: {self.score}"
        text_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (620 - text_surf.get_width(), 15))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self._check_win_condition():
                msg = "Puzzle Solved!"
            else:
                msg = "Out of Moves!"
            
            msg_surf = self.font_large.render(msg, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(320, 200))
            self.screen.blit(msg_surf, msg_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a dummy screen for display since render_mode is "rgb_array"
    pygame.display.set_caption("Isometric Box Puzzler")
    display_screen = pygame.display.set_mode((640, 400))
    
    action = [0, 0, 0] # No-op, space released, shift released
    
    running = True
    while running:
        # Human input handling
        space_pressed_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: 
                    action[1] = 1
                    space_pressed_this_frame = True # Flag to consume action
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    done = False
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        # Only step if an action is taken
        if action[0] != 0 or space_pressed_this_frame:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Moves: {info['moves_remaining']}")
            
            # Reset movement action after it's processed for turn-based play
            action[0] = 0 
        
        # Update the display
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS

    pygame.quit()