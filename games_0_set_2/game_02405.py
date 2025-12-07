
# Generated: 2025-08-28T04:41:21.635156
# Source Brief: brief_02405.md
# Brief Index: 2405

        
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
    A grid-based puzzle game where the player pushes boxes along one-way paths
    to their designated goals within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to apply a push force to all boxes simultaneously. "
        "Boxes only move in valid directions, respecting walls and one-way path arrows."
    )

    # Short, user-facing description of the game
    game_description = (
        "A turn-based puzzle game. Push all the boxes onto their matching goals. "
        "Each push costs one move. Plan your moves carefully as they are irreversible!"
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 10
        self.CELL_SIZE = 40
        self.GRID_AREA_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.UI_AREA_WIDTH = self.WIDTH - self.GRID_AREA_WIDTH
        self.MAX_MOVES = 15
        self.NUM_BOXES = 3

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (50, 55, 65)
        self.COLOR_UI_BG = (35, 38, 48)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_TEXT_GOOD = (138, 212, 157)
        self.COLOR_UI_TEXT_BAD = (212, 138, 138)
        self.COLOR_BOX = (199, 111, 71)
        self.COLOR_BOX_LIT = (222, 148, 106)
        self.COLOR_GOAL = (80, 158, 129)
        self.COLOR_GOAL_LIT = (138, 212, 157)
        self.COLOR_ARROW = (78, 128, 194)
        self.COLOR_WIN = (138, 212, 157, 200)
        self.COLOR_LOSE = (212, 138, 138, 200)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 14)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.moves_left = 0
        self.game_over = False
        self.win_condition = False
        
        self.box_positions = []
        self.goal_positions = []
        self.one_way_paths = {}
        self.last_move_info = None

        # --- Initialize state and validate ---
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win_condition = False
        self.last_move_info = None

        # --- Generate a new puzzle ---
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_cells)
        
        # Place goals
        self.goal_positions = all_cells[:self.NUM_BOXES]
        
        # Place boxes, ensuring they are not on goals
        valid_box_cells = [cell for cell in all_cells if cell not in self.goal_positions]
        self.box_positions = valid_box_cells[:self.NUM_BOXES]
        
        # Generate one-way paths (on ~30% of non-goal cells)
        self.one_way_paths = {}
        path_candidates = [cell for cell in all_cells if cell not in self.goal_positions]
        num_paths = int(len(path_candidates) * 0.3)
        path_cells = self.np_random.choice(len(path_candidates), num_paths, replace=False)
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for i in path_cells:
            cell = path_candidates[i]
            direction = directions[self.np_random.integers(0, 4)]
            self.one_way_paths[cell] = direction

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        # No-op action
        if movement == 0:
            return self._get_observation(), 0, False, False, self._get_info()

        # --- Execute Move ---
        self.steps += 1
        self.moves_left -= 1
        reward = -0.1  # Cost for making a move
        
        # Store initial goal status
        boxes_on_goal_before = self._count_boxes_on_goals()

        # --- Process Box Movement ---
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = move_map[movement]
        
        # Sort boxes to handle chain pushes correctly
        sorted_indices = sorted(
            range(len(self.box_positions)),
            key=lambda i: (self.box_positions[i][0] * dx + self.box_positions[i][1] * dy),
            reverse=True
        )
        
        new_positions = list(self.box_positions)
        moved_boxes_indices = []

        for i in sorted_indices:
            bx, by = new_positions[i]
            
            # Check one-way path constraint
            if (bx, by) in self.one_way_paths and self.one_way_paths[(bx, by)] != (dx, dy):
                continue

            nx, ny = bx + dx, by + dy
            
            # Check boundaries
            if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
                continue
            
            # Check collision with other boxes
            if (nx, ny) in new_positions:
                continue
            
            # Valid move, update position
            new_positions[i] = (nx, ny)
            moved_boxes_indices.append(i)
        
        self.last_move_info = {
            'moved_indices': moved_boxes_indices,
            'direction': (dx, dy),
            'old_positions': self.box_positions
        }
        self.box_positions = new_positions

        # --- Calculate Rewards and Termination ---
        boxes_on_goal_after = self._count_boxes_on_goals()
        newly_on_goal = boxes_on_goal_after - boxes_on_goal_before
        
        if newly_on_goal > 0:
            reward += newly_on_goal * 1.0  # Reward for placing a box
            
        self.score += reward
        
        self.win_condition = (boxes_on_goal_after == self.NUM_BOXES)
        
        terminated = self.win_condition or self.moves_left <= 0
        self.game_over = terminated

        if terminated:
            if self.win_condition:
                terminal_reward = 10.0
                self.score += terminal_reward
                reward += terminal_reward
            else: # Out of moves
                terminal_reward = -10.0
                self.score += terminal_reward
                reward += terminal_reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Clear last move info after rendering it once
        self.last_move_info = None

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.GRID_AREA_WIDTH, py))
            
        # Draw one-way arrows
        for (x, y), (dx, dy) in self.one_way_paths.items():
            self._draw_arrow(x, y, dx, dy)

        # Draw goals
        for gx, gy in self.goal_positions:
            is_occupied = (gx, gy) in self.box_positions
            color = self.COLOR_GOAL_LIT if is_occupied else self.COLOR_GOAL
            center_x = int((gx + 0.5) * self.CELL_SIZE)
            center_y = int((gy + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Draw boxes
        for i, (bx, by) in enumerate(self.box_positions):
            rect = pygame.Rect(
                bx * self.CELL_SIZE + 4, by * self.CELL_SIZE + 4,
                self.CELL_SIZE - 8, self.CELL_SIZE - 8
            )
            is_on_goal = (bx, by) in self.goal_positions
            color = self.COLOR_BOX_LIT if is_on_goal else self.COLOR_BOX
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, width=2, border_radius=4)

    def _draw_arrow(self, x, y, dx, dy):
        center_x = (x + 0.5) * self.CELL_SIZE
        center_y = (y + 0.5) * self.CELL_SIZE
        size = self.CELL_SIZE * 0.15

        if dx == 1: # Right
            p1 = (center_x - size, center_y - size)
            p2 = (center_x + size, center_y)
            p3 = (center_x - size, center_y + size)
        elif dx == -1: # Left
            p1 = (center_x + size, center_y - size)
            p2 = (center_x - size, center_y)
            p3 = (center_x + size, center_y + size)
        elif dy == 1: # Down
            p1 = (center_x - size, center_y - size)
            p2 = (center_x, center_y + size)
            p3 = (center_x + size, center_y - size)
        else: # Up
            p1 = (center_x - size, center_y + size)
            p2 = (center_x, center_y - size)
            p3 = (center_x + size, center_y + size)
        
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ARROW)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ARROW)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(self.GRID_AREA_WIDTH, 0, self.UI_AREA_WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        
        # Title
        title_surf = self.font_main.render("SOKOBAN", True, self.COLOR_UI_TEXT)
        self.screen.blit(title_surf, (self.GRID_AREA_WIDTH + 25, 20))
        
        # Moves Left
        moves_label = self.font_small.render("MOVES LEFT", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_label, (self.GRID_AREA_WIDTH + 25, 70))
        moves_val_color = self.COLOR_UI_TEXT_BAD if self.moves_left <= 5 else self.COLOR_UI_TEXT_GOOD
        moves_val = self.font_main.render(str(self.moves_left), True, moves_val_color)
        self.screen.blit(moves_val, (self.GRID_AREA_WIDTH + 25, 90))

        # Score
        score_label = self.font_small.render("SCORE", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_label, (self.GRID_AREA_WIDTH + 25, 140))
        score_val = self.font_main.render(f"{self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_val, (self.GRID_AREA_WIDTH + 25, 160))
        
        # Goal status
        goal_label = self.font_small.render("GOALS", True, self.COLOR_UI_TEXT)
        self.screen.blit(goal_label, (self.GRID_AREA_WIDTH + 25, 210))
        
        boxes_on_goal = self._count_boxes_on_goals()
        goal_status_text = f"{boxes_on_goal} / {self.NUM_BOXES}"
        goal_status_color = self.COLOR_UI_TEXT_GOOD if boxes_on_goal == self.NUM_BOXES else self.COLOR_UI_TEXT
        goal_val = self.font_main.render(goal_status_text, True, goal_status_color)
        self.screen.blit(goal_val, (self.GRID_AREA_WIDTH + 25, 230))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.GRID_AREA_WIDTH, self.HEIGHT), pygame.SRCALPHA)
            if self.win_condition:
                text = "PUZZLE SOLVED!"
                color = self.COLOR_WIN
            else:
                text = "OUT OF MOVES"
                color = self.COLOR_LOSE
            
            overlay.fill(color)
            self.screen.blit(overlay, (0, 0))
            
            text_surf = self.font_large.render(text, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.GRID_AREA_WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "boxes_on_goal": self._count_boxes_on_goals(),
        }

    def _count_boxes_on_goals(self):
        return sum(1 for box_pos in self.box_positions if box_pos in self.goal_positions)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sokoban Paths")
    clock = pygame.time.Clock()
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                    continue
                elif event.key == pygame.K_q:
                    running = False
                    
                # Take a step if an action was taken
                if action[0] != 0 and not done:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS

    env.close()