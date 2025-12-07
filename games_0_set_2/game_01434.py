# Generated: 2025-08-27T17:07:12.613207
# Source Brief: brief_01434.md
# Brief Index: 1434

        
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
        "Controls: Use arrow keys (↑↓←→) to move your character. Push all the colored boxes onto their matching targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic block-pushing puzzle. Strategically move boxes onto their designated targets within the move limit to advance through stages."
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
        
        # Visuals & Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_WALL = (100, 110, 130)
        self.COLOR_WALL_BORDER = (80, 90, 110)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_BORDER = (40, 120, 200)
        self.COLOR_TEXT = (220, 220, 240)
        
        self.BOX_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
            (255, 160, 80),  # Orange
            (160, 80, 255),  # Purple
            (200, 200, 200)  # White
        ]
        # FIX: Convert generator expression to a tuple to create a valid color argument.
        self.TARGET_COLORS = [tuple(min(255, c + 30) for c in color) for color in self.BOX_COLORS]
        self.TARGET_MARK_COLOR = (255, 255, 255, 100) # Semi-transparent white for 'X'

        # Fonts
        self.font_large = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 18)
        
        # Game state variables
        self.player_pos = (0, 0)
        self.boxes = []
        self.targets = []
        self.walls = []
        self.grid_dims = (0, 0)
        self.tile_size = 0
        self.grid_offset = (0, 0)
        self.current_stage = 0
        self.moves_left = 0
        self.stage_configs = self._define_stages()
        
        # Initialize state
        # The reset call is deferred to the first call to reset() as per Gymnasium API.
        # self.reset() is not called in __init__ anymore to allow proper seeding.
        self.steps = 0
        self.score = 0
        self.game_over = True # Start in a terminal state, reset will start the game.

    def _define_stages(self):
        return {
            1: {
                "grid": (10, 8),
                "moves": 25,
                "player": (1, 1),
                "boxes": [(3, 3), (4, 5), (6, 2)],
                "targets": [(8, 6), (1, 4), (5, 4)],
                "walls": []
            },
            2: {
                "grid": (12, 10),
                "moves": 40,
                "player": (5, 1),
                "boxes": [(3, 3), (4, 6), (7, 4), (8, 8)],
                "targets": [(1, 8), (10, 2), (10, 8), (1, 2)],
                "walls": []
            },
            3: {
                "grid": (14, 11),
                "moves": 60,
                "player": (1, 1),
                "boxes": [(4, 2), (5, 5), (7, 3), (9, 6), (9, 8)],
                "targets": [(12, 9), (1, 9), (6, 9), (1, 5), (6, 1)],
                "walls": list(set(
                    [(r, 0) for r in range(14)] + [(r, 10) for r in range(14)] +
                    [(0, c) for c in range(11)] + [(13, c) for c in range(11)] +
                    [(r, 4) for r in range(3, 11)] + [(r, 7) for r in range(2, 9)] +
                    [(3, c) for c in range(1, 4)] + [(10, c) for c in range(5, 9)]
                ))
            }
        }

    def _load_stage(self, stage_num):
        if stage_num not in self.stage_configs:
            self.game_over = True
            return

        config = self.stage_configs[stage_num]
        self.grid_dims = config["grid"]
        self.moves_left = config["moves"]
        self.player_pos = config["player"]
        self.boxes = [list(pos) for pos in config["boxes"]]
        self.targets = [list(pos) for pos in config["targets"]]
        self.walls = [tuple(pos) for pos in config["walls"]]
        self.current_stage = stage_num

        # Calculate rendering geometry
        grid_w, grid_h = self.grid_dims
        self.tile_size = min(
            (self.screen.get_width() - 80) // (grid_w if grid_w > 0 else 1),
            (self.screen.get_height() - 80) // (grid_h if grid_h > 0 else 1)
        )
        total_grid_width = self.tile_size * grid_w
        total_grid_height = self.tile_size * grid_h
        self.grid_offset = (
            (self.screen.get_width() - total_grid_width) // 2,
            (self.screen.get_height() - total_grid_height) // 2
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._load_stage(1)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        
        if movement == 0: # No-op
            self.steps += 1
            return self._get_observation(), 0, False, False, self._get_info()
        
        # --- Pre-move state for reward calculation ---
        old_distances = self._get_total_manhattan_distance()
        old_on_target_count = self._get_on_target_count()

        # --- Game Logic ---
        self.steps += 1
        self.moves_left -= 1
        # sound: player_move.wav
        
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # (dc, dr) -> (dx, dy)
        if movement == 1: # Up
            dr, dc = -1, 0
        elif movement == 2: # Down
            dr, dc = 1, 0
        elif movement == 3: # Left
            dr, dc = 0, -1
        elif movement == 4: # Right
            dr, dc = 0, 1
        else:
             dr, dc = 0, 0
        
        player_r, player_c = self.player_pos
        next_r, next_c = player_r + dr, player_c + dc
        
        # Check boundaries and walls
        if not (0 <= next_c < self.grid_dims[1] and 0 <= next_r < self.grid_dims[0]) or (next_r, next_c) in self.walls:
            # Player walks into wall, move fails
            pass # sound: bump_wall.wav
        # Check for box push
        elif [next_r, next_c] in self.boxes:
            box_idx = self.boxes.index([next_r, next_c])
            next_box_r, next_box_c = next_r + dr, next_c + dc
            
            # Check if space behind box is clear
            if (0 <= next_box_c < self.grid_dims[1] and 0 <= next_box_r < self.grid_dims[0] and
                [next_box_r, next_box_c] not in self.boxes and (next_box_r, next_box_c) not in self.walls):
                # Push box
                self.boxes[box_idx] = [next_box_r, next_box_c]
                self.player_pos = (next_r, next_c)
                # sound: push_box.wav
            else:
                # Box is blocked
                pass # sound: bump_box.wav
        else:
            # Simple move
            self.player_pos = (next_r, next_c)

        # --- Post-move reward and state update ---
        new_distances = self._get_total_manhattan_distance()
        new_on_target_count = self._get_on_target_count()
        
        # Reward for moving boxes closer to targets
        reward += (old_distances - new_distances) * 0.1
        
        # Reward for placing a box on a target
        if new_on_target_count > old_on_target_count:
            reward += (new_on_target_count - old_on_target_count) * 5
            # sound: box_on_target.wav

        self.score += reward
        
        # Stage completion rewards
        if self._check_stage_complete() and not self.game_over:
            if self.current_stage < 3:
                reward += 50
                self.score += 50
                # sound: stage_clear.wav
                self._load_stage(self.current_stage + 1)
            else: # Final stage completed
                reward += 100
                self.score += 100
                self.game_over = True
                # sound: game_win.wav

        if not self.game_over and self.moves_left <= 0:
            self.game_over = True
            # sound: game_over.wav

        if self.steps >= 1000:
            self.game_over = True

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_total_manhattan_distance(self):
        dist = 0
        # Create a copy of targets to avoid modifying the original list
        unmatched_targets = list(self.targets)
        for box_pos in self.boxes:
            # Find the closest unmatched target for the current box
            closest_dist = float('inf')
            best_target = None
            for target_pos in unmatched_targets:
                d = abs(box_pos[0] - target_pos[0]) + abs(box_pos[1] - target_pos[1])
                if d < closest_dist:
                    closest_dist = d
                    best_target = target_pos
            if best_target is not None:
                dist += closest_dist
                unmatched_targets.remove(best_target)
        return dist

    def _get_on_target_count(self):
        count = 0
        # A box is on target if its position matches any of the target positions
        box_positions = [tuple(b) for b in self.boxes]
        target_positions = [tuple(t) for t in self.targets]
        for box_pos in box_positions:
            if box_pos in target_positions:
                count += 1
        return count

    def _check_stage_complete(self):
        # Stage is complete if the set of box positions is the same as the set of target positions
        return set(map(tuple, self.boxes)) == set(map(tuple, self.targets))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        if not self.game_over:
            self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "moves_left": self.moves_left,
        }

    def _render_game(self):
        # Draw grid lines
        grid_w, grid_h = self.grid_dims
        for r in range(grid_h + 1):
            y = self.grid_offset[1] + r * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset[0], y), (self.grid_offset[0] + grid_w * self.tile_size, y))
        for c in range(grid_w + 1):
            x = self.grid_offset[0] + c * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset[1]), (x, self.grid_offset[1] + grid_h * self.tile_size))

        # Draw targets
        box_tuples = set(map(tuple, self.boxes))
        for i, pos in enumerate(self.targets):
            r, c = pos
            is_covered = tuple(pos) in box_tuples
            
            center_x = self.grid_offset[0] + int((c + 0.5) * self.tile_size)
            center_y = self.grid_offset[1] + int((r + 0.5) * self.tile_size)
            radius = int(self.tile_size * 0.35)
            
            target_color = self.TARGET_COLORS[i % len(self.TARGET_COLORS)]
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, target_color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, target_color)

            if not is_covered:
                # Draw an 'X'
                d = int(self.tile_size * 0.2)
                # Use a color without alpha for pygame.draw.line on a non-alpha surface
                mark_color_rgb = self.TARGET_MARK_COLOR[:3]
                pygame.draw.line(self.screen, mark_color_rgb, (center_x - d, center_y - d), (center_x + d, center_y + d), 3)
                pygame.draw.line(self.screen, mark_color_rgb, (center_x - d, center_y + d), (center_x + d, center_y - d), 3)

        # Draw walls
        for r, c in self.walls:
            rect = pygame.Rect(
                self.grid_offset[0] + c * self.tile_size,
                self.grid_offset[1] + r * self.tile_size,
                self.tile_size, self.tile_size
            )
            pygame.draw.rect(self.screen, self.COLOR_WALL_BORDER, rect)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect.inflate(-4, -4))

        # Draw boxes
        for i, pos in enumerate(self.boxes):
            r, c = pos
            rect = pygame.Rect(
                self.grid_offset[0] + c * self.tile_size + 2,
                self.grid_offset[1] + r * self.tile_size + 2,
                self.tile_size - 4, self.tile_size - 4
            )
            box_color = self.BOX_COLORS[i % len(self.BOX_COLORS)]
            border_color = tuple(max(0, val-40) for val in box_color)
            pygame.draw.rect(self.screen, border_color, rect, 0, 4)
            inner_rect = rect.inflate(-6, -6)
            pygame.draw.rect(self.screen, box_color, inner_rect, 0, 3)

        # Draw player
        r, c = self.player_pos
        rect = pygame.Rect(
            self.grid_offset[0] + c * self.tile_size + 2,
            self.grid_offset[1] + r * self.tile_size + 2,
            self.tile_size - 4, self.tile_size - 4
        )
        # Simple breathing animation
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
        glow_size = int(pulse * 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_BORDER, rect.inflate(glow_size, glow_size), 0, 5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, 0, 4)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))

        # Stage
        stage_text = self.font_large.render(f"Stage: {self.current_stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.screen.get_width() - stage_text.get_width() - 20, 10))

        # Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen.get_width() // 2 - score_text.get_width() // 2, 15))
        
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._check_stage_complete() and self.current_stage == 3:
                end_text_str = "YOU WIN!"
            else:
                end_text_str = "GAME OVER"
                
            end_text = self.font_large.render(end_text_str, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()