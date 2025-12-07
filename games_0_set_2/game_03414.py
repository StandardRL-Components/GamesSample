
# Generated: 2025-08-27T23:16:58.425558
# Source Brief: brief_03414.md
# Brief Index: 3414

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selected shape. Press space to cycle which shape is selected."
    )

    game_description = (
        "A minimalist puzzle game. Move the colored shapes to fit perfectly into their matching empty outlines. You have a limited number of moves to solve the puzzle!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    MAX_MOVES = 15
    MAX_STEPS = 1000
    NUM_SHAPES = 8

    # --- Colors ---
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (45, 45, 60)
    COLOR_OUTLINE = (120, 40, 40)
    COLOR_FILLED = (40, 180, 100)
    COLOR_SHAPE_INACTIVE = (120, 120, 140)
    COLOR_SHAPE_SELECTED = (60, 160, 255)
    COLOR_SHAPE_SELECTED_GLOW = (60, 160, 255, 64)
    COLOR_TEXT = (220, 220, 240)
    COLOR_WIN = (100, 255, 150)
    COLOR_LOSE = (255, 100, 100)

    # --- Shape Definitions (relative (x, y) coordinates) ---
    SHAPE_DEFS = {
        0: [(0, 0), (1, 0), (2, 0), (3, 0)],  # I
        1: [(0, 0), (1, 0), (0, 1), (1, 1)],  # O
        2: [(1, 0), (0, 1), (1, 1), (2, 1)],  # T
        3: [(0, 0), (0, 1), (0, 2), (1, 2)],  # L
        4: [(1, 0), (1, 1), (1, 2), (0, 2)],  # J
        5: [(1, 0), (2, 0), (0, 1), (1, 1)],  # S
        6: [(0, 0), (1, 0), (1, 1), (2, 1)],  # Z
        7: [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)], # Plus
    }

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
        self.font_ui = pygame.font.Font(None, 32)
        self.font_msg = pygame.font.Font(None, 64)
        
        self.shapes = []
        self.outlines = []
        self.unsolved_indices = []
        self.selected_unsolved_idx = 0
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.end_message = ""
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.end_message = ""

        self._place_entities()
        self.unsolved_indices = [i for i, shape in enumerate(self.shapes) if not shape['solved']]
        self.selected_unsolved_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Handle selection change
        if space_press and self.unsolved_indices:
            self.selected_unsolved_idx = (self.selected_unsolved_idx + 1) % len(self.unsolved_indices)
            # sfx: selection_change.wav

        # 2. Handle movement
        moved = False
        if movement != 0 and self.moves_left > 0 and self.unsolved_indices:
            self.moves_left -= 1
            moved = True
            
            shape_idx = self.unsolved_indices[self.selected_unsolved_idx]
            shape = self.shapes[shape_idx]
            outline = self.outlines[shape_idx]
            
            dx = {3: -1, 4: 1}.get(movement, 0)
            dy = {1: -1, 2: 1}.get(movement, 0)
            
            old_pos = shape['pos']
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            
            dist_before = math.hypot(old_pos[0] - outline['pos'][0], old_pos[1] - outline['pos'][1])

            if self._is_valid_move(shape_idx, new_pos):
                shape['pos'] = new_pos
                dist_after = math.hypot(new_pos[0] - outline['pos'][0], new_pos[1] - outline['pos'][1])
                
                if dist_after < dist_before:
                    reward += 1
                elif dist_after > dist_before:
                    reward -= 1
                # sfx: move_shape.wav
            else:
                # sfx: move_blocked.wav
                pass # Invalid move, no change

        # 3. Check for solved pieces
        newly_solved_count = 0
        temp_unsolved = []
        for i in self.unsolved_indices:
            shape = self.shapes[i]
            outline = self.outlines[i]
            if shape['pos'] == outline['pos']:
                shape['solved'] = True
                self.score += 10
                reward += 10
                newly_solved_count += 1
                # sfx: shape_solved.wav
            else:
                temp_unsolved.append(i)
        
        if newly_solved_count > 0:
            self.unsolved_indices = temp_unsolved
            if self.unsolved_indices: # If there are still shapes left to solve
                self.selected_unsolved_idx = self.selected_unsolved_idx % len(self.unsolved_indices)

        # 4. Update game state and check for termination
        self.steps += 1
        terminated = False
        
        if not self.unsolved_indices:
            self.game_over = True
            terminated = True
            self.score += 50
            reward += 50
            self.end_message = "YOU WIN!"
            # sfx: game_win.wav
        elif self.moves_left <= 0 and moved:
            self.game_over = True
            terminated = True
            self.end_message = "OUT OF MOVES"
            # sfx: game_lose.wav
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            self.end_message = "TIME UP"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_outlines()
        self._draw_shapes()
        self._render_ui()
        if self.game_over:
            self._render_end_message()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _place_entities(self):
        self.shapes.clear()
        self.outlines.clear()
        
        shape_ids = list(range(self.NUM_SHAPES))
        self.np_random.shuffle(shape_ids)
        
        occupied_cells = set()

        # Place outlines
        for i in range(self.NUM_SHAPES):
            shape_id = shape_ids[i]
            shape_def = self.SHAPE_DEFS[shape_id]
            
            for _ in range(100): # Max 100 placement attempts
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                abs_coords = {(pos[0] + dx, pos[1] + dy) for dx, dy in shape_def}
                
                in_bounds = all(0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT for x, y in abs_coords)
                overlaps = any(cell in occupied_cells for cell in abs_coords)
                
                if in_bounds and not overlaps:
                    self.outlines.append({'pos': pos, 'shape_id': shape_id})
                    occupied_cells.update(abs_coords)
                    break
        
        # Place shapes
        for i in range(self.NUM_SHAPES):
            shape_id = self.outlines[i]['shape_id']
            shape_def = self.SHAPE_DEFS[shape_id]
            
            for _ in range(100):
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                abs_coords = {(pos[0] + dx, pos[1] + dy) for dx, dy in shape_def}

                in_bounds = all(0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT for x, y in abs_coords)
                overlaps = any(cell in occupied_cells for cell in abs_coords)

                if in_bounds and not overlaps:
                    self.shapes.append({'pos': pos, 'shape_id': shape_id, 'solved': False})
                    occupied_cells.update(abs_coords)
                    break

    def _is_valid_move(self, moving_shape_idx, new_pos):
        shape_def = self.SHAPE_DEFS[self.shapes[moving_shape_idx]['shape_id']]
        
        # Check boundaries
        for dx, dy in shape_def:
            nx, ny = new_pos[0] + dx, new_pos[1] + dy
            if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
                return False
        
        # Check collision with other shapes
        for i, other_shape in enumerate(self.shapes):
            if i == moving_shape_idx or other_shape['solved']:
                continue
            
            other_shape_def = self.SHAPE_DEFS[other_shape['shape_id']]
            for dx1, dy1 in shape_def:
                for dx2, dy2 in other_shape_def:
                    if (new_pos[0] + dx1, new_pos[1] + dy1) == (other_shape['pos'][0] + dx2, other_shape['pos'][1] + dy2):
                        return False
        return True

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_outlines(self):
        for i, outline in enumerate(self.outlines):
            is_filled = self.shapes[i]['solved']
            color = self.COLOR_FILLED if is_filled else self.COLOR_OUTLINE
            self._draw_component(outline['pos'], outline['shape_id'], color, is_outline=True)

    def _draw_shapes(self):
        selected_idx = self.unsolved_indices[self.selected_unsolved_idx] if self.unsolved_indices else -1
        
        for i, shape in enumerate(self.shapes):
            if shape['solved']:
                # Draw as filled, matching the outline
                self._draw_component(shape['pos'], shape['shape_id'], self.COLOR_FILLED, is_outline=False)
            else:
                is_selected = (i == selected_idx)
                color = self.COLOR_SHAPE_SELECTED if is_selected else self.COLOR_SHAPE_INACTIVE
                self._draw_component(shape['pos'], shape['shape_id'], color, is_outline=False, is_selected=is_selected)

    def _draw_component(self, grid_pos, shape_id, color, is_outline, is_selected=False):
        shape_def = self.SHAPE_DEFS[shape_id]
        px, py = grid_pos[0] * self.CELL_SIZE, grid_pos[1] * self.CELL_SIZE
        
        if is_selected:
            for dx, dy in shape_def:
                glow_rect = pygame.Rect(
                    px + dx * self.CELL_SIZE - 3, 
                    py + dy * self.CELL_SIZE - 3,
                    self.CELL_SIZE + 6, self.CELL_SIZE + 6
                )
                pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_SHAPE_SELECTED_GLOW)

        for dx, dy in shape_def:
            rect = pygame.Rect(
                px + dx * self.CELL_SIZE, 
                py + dy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            if is_outline:
                pygame.draw.rect(self.screen, color, rect, width=2, border_radius=3)
            else:
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, width=2, border_radius=3) # Inner border

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

    def _render_end_message(self):
        color = self.COLOR_WIN if "WIN" in self.end_message else self.COLOR_LOSE
        msg_surf = self.font_msg.render(self.end_message, True, color)
        msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        # Draw a semi-transparent background for the message
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        self.screen.blit(msg_surf, msg_rect)

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `SDL_VIDEODRIVER=dummy` in your environment if you are running headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen_human = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Shifter")
    clock = pygame.time.Clock()
    running = True
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # Game only advances on an action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        # Render the observation from the environment to the human-facing screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(10) # Run at 10 FPS for human play

    env.close()