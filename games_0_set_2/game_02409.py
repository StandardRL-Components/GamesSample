
# Generated: 2025-08-28T04:44:44.419894
# Source Brief: brief_02409.md
# Brief Index: 2409

        
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
        "Controls: Arrow keys to push boxes relative to the platform. Hold shift to rotate the platform clockwise."
    )

    game_description = (
        "A minimalist puzzle game. Push the red boxes onto the green targets by moving and rotating the platform. You have a limited number of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.CELL_SIZE = 50
        self.PLATFORM_PIXEL_SIZE = self.GRID_SIZE * self.CELL_SIZE
        self.MAX_MOVES = 25

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PLATFORM = (50, 55, 60)
        self.COLOR_GRID = (70, 75, 80)
        self.COLOR_BOX = (220, 50, 50)
        self.COLOR_BOX_LIT = (255, 100, 100)
        self.COLOR_TARGET = (50, 200, 50)
        self.COLOR_TARGET_LIT = (100, 255, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_ARROW = (255, 255, 0)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.moves = 0
        self.score = 0
        self.game_over = False
        self.rotation = 0  # 0: 0, 1: 90, 2: 180, 3: 270 degrees
        self.box_positions = []
        self.target_positions = []
        self.target_tuples = set()
        self.boxes_on_target_before_step = set()
        self.last_push_direction = None
        self.last_action_was_rotation = False
        self.np_random = None

        # Pre-defined levels
        self.levels = [
            {"boxes": [(1, 1), (3, 3)], "targets": [(0, 2), (4, 2)]},
            {"boxes": [(0, 0), (4, 4)], "targets": [(2, 0), (2, 4)]},
            {"boxes": [(1, 2), (3, 2)], "targets": [(2, 1), (2, 3)]},
            {"boxes": [(0, 1), (0, 3)], "targets": [(4, 1), (4, 3)]},
            {"boxes": [(2, 0), (2, 4)], "targets": [(0, 2), (4, 2)]},
        ]
        
        # World-space directions for push actions
        self.DIRECTIONS = {
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0),   # Right
        }

        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset game state
        self.steps = 0
        self.moves = 0
        self.score = 0
        self.game_over = False
        self.rotation = 0
        self.last_push_direction = None
        self.last_action_was_rotation = False

        # Select a level randomly
        level_idx = self.np_random.integers(0, len(self.levels))
        level = self.levels[level_idx]
        self.box_positions = [list(p) for p in level["boxes"]]
        self.target_positions = [list(p) for p in level["targets"]]
        self.target_tuples = {tuple(p) for p in self.target_positions}

        self.boxes_on_target_before_step = self._get_boxes_on_target()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        action_taken = False
        self.last_push_direction = None
        self.last_action_was_rotation = False

        movement, _, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Action Handling ---
        # Prioritize rotation over push
        if shift_held:
            # SFX: whoosh_rotate.wav
            self.rotation = (self.rotation + 1) % 4
            action_taken = True
            self.last_action_was_rotation = True
        elif movement != 0:
            world_direction = self._get_world_direction(movement)
            if self._handle_push(world_direction):
                # SFX: slide_box.wav
                action_taken = True
                self.last_push_direction = world_direction
            else:
                # SFX: bump_wall.wav
                pass

        # --- State and Reward Update ---
        if action_taken:
            self.moves += 1
            reward -= 0.1  # Cost per move

            # Calculate reward for boxes on/off targets
            boxes_on_target_after_step = self._get_boxes_on_target()
            newly_on_target = boxes_on_target_after_step - self.boxes_on_target_before_step
            newly_off_target = self.boxes_on_target_before_step - boxes_on_target_after_step
            
            if newly_on_target:
                # SFX: success_ding.wav
                reward += len(newly_on_target) * 5.0
            if newly_off_target:
                # SFX: error_buzz.wav
                reward += len(newly_off_target) * -1.0
            
            self.boxes_on_target_before_step = boxes_on_target_after_step

        self.score += reward
        self.steps += 1

        # --- Termination Check ---
        won = len(self.boxes_on_target_before_step) == len(self.box_positions)
        lost = self.moves >= self.MAX_MOVES
        terminated = won or lost
        
        if terminated:
            self.game_over = True
            if won:
                # SFX: victory_fanfare.wav
                self.score += 100
                reward += 100
            else: # lost
                # SFX: failure_trombone.wav
                self.score -= 100
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_world_direction(self, movement_action):
        """Converts a relative movement action to a world-space direction vector."""
        ax, ay = self.DIRECTIONS[movement_action]
        # Apply rotation to the action vector
        # 0 deg: (x, y) -> (x, y)
        # 90 deg: (x, y) -> (y, -x)
        # 180 deg: (x, y) -> (-x, -y)
        # 270 deg: (x, y) -> (-y, x)
        if self.rotation == 1: (ax, ay) = (ay, -ax)
        elif self.rotation == 2: (ax, ay) = (-ax, -ay)
        elif self.rotation == 3: (ax, ay) = (-ay, ax)
        return (ax, ay)

    def _handle_push(self, world_direction):
        """Attempts to push all movable boxes in the given world direction."""
        dx, dy = world_direction
        moved_any_box = False
        
        box_indices = list(range(len(self.box_positions)))
        # Sort boxes to handle pushes correctly (from the "back" of the push direction)
        if dx > 0: box_indices.sort(key=lambda i: self.box_positions[i][0], reverse=True)
        elif dx < 0: box_indices.sort(key=lambda i: self.box_positions[i][0])
        if dy > 0: box_indices.sort(key=lambda i: self.box_positions[i][1], reverse=True)
        elif dy < 0: box_indices.sort(key=lambda i: self.box_positions[i][1])

        for i in box_indices:
            pos = self.box_positions[i]
            next_pos = [pos[0] + dx, pos[1] + dy]
            
            # Check bounds
            if not (0 <= next_pos[0] < self.GRID_SIZE and 0 <= next_pos[1] < self.GRID_SIZE):
                continue

            # Check collision with other boxes
            if any(other_pos == next_pos for other_pos in self.box_positions):
                continue
            
            self.box_positions[i] = next_pos
            moved_any_box = True
            
        return moved_any_box

    def _get_boxes_on_target(self):
        """Returns a set of indices for boxes currently on a target."""
        return {i for i, pos in enumerate(self.box_positions) if tuple(pos) in self.target_tuples}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        platform_center = (self.WIDTH // 2, self.HEIGHT // 2)
        angle_rad = math.radians(self.rotation * 90)

        # Draw platform base
        platform_rect = pygame.Rect(0, 0, self.PLATFORM_PIXEL_SIZE, self.PLATFORM_PIXEL_SIZE)
        platform_rect.center = platform_center
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform_rect)

        # Helper to convert grid coords to rotated world coords
        def get_rotated_world_pos(grid_x, grid_y, offset_x=0, offset_y=0):
            # 1. To local pixel coords relative to platform center
            local_x = (grid_x - self.GRID_SIZE / 2 + 0.5) * self.CELL_SIZE + offset_x
            local_y = (grid_y - self.GRID_SIZE / 2 + 0.5) * self.CELL_SIZE + offset_y
            # 2. Rotate
            rotated_x = local_x * math.cos(angle_rad) - local_y * math.sin(angle_rad)
            rotated_y = local_x * math.sin(angle_rad) + local_y * math.cos(angle_rad)
            # 3. To world coords
            return (int(platform_center[0] + rotated_x), int(platform_center[1] + rotated_y))

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start = get_rotated_world_pos(i, 0, offset_x=-self.CELL_SIZE/2)
            end = get_rotated_world_pos(i, self.GRID_SIZE, offset_x=-self.CELL_SIZE/2)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
            # Horizontal lines
            start = get_rotated_world_pos(0, i, offset_y=-self.CELL_SIZE/2)
            end = get_rotated_world_pos(self.GRID_SIZE, i, offset_y=-self.CELL_SIZE/2)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        
        # Draw targets
        for gx, gy in self.target_positions:
            pos = get_rotated_world_pos(gx, gy)
            is_occupied = tuple([gx, gy]) in [tuple(bp) for bp in self.box_positions]
            color = self.COLOR_TARGET_LIT if is_occupied else self.COLOR_TARGET
            radius = int(self.CELL_SIZE * 0.35)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

        # Draw boxes
        box_half_size = self.CELL_SIZE * 0.4
        for i, (gx, gy) in enumerate(self.box_positions):
            is_on_target = i in self.boxes_on_target_before_step
            color = self.COLOR_BOX_LIT if is_on_target else self.COLOR_BOX
            
            # Define 4 corners of the square in local space
            corners = [
                (-box_half_size, -box_half_size), (box_half_size, -box_half_size),
                (box_half_size, box_half_size), (-box_half_size, box_half_size)
            ]
            # Rotate and translate corners
            rotated_corners = []
            for corner_x, corner_y in corners:
                pos = get_rotated_world_pos(gx, gy, offset_x=corner_x, offset_y=corner_y)
                rotated_corners.append(pos)
            
            pygame.gfxdraw.filled_polygon(self.screen, rotated_corners, color)
            pygame.gfxdraw.aapolygon(self.screen, rotated_corners, color)

        # Draw action indicator
        if self.last_action_was_rotation:
            self._draw_rotation_indicator(platform_center)
        elif self.last_push_direction:
            self._draw_push_indicator(platform_center, self.last_push_direction)
            
    def _draw_rotation_indicator(self, center):
        radius = 20
        # Draw a curved arrow
        rect = pygame.Rect(center[0] - radius, center[1] - radius, radius*2, radius*2)
        pygame.draw.arc(self.screen, self.COLOR_ARROW, rect, math.radians(30), math.radians(300), 3)
        # Arrowhead
        p1 = (center[0] + radius * math.cos(math.radians(30)), center[1] + radius * math.sin(math.radians(30)))
        p2 = (p1[0] - 10, p1[1] - 5)
        p3 = (p1[0] - 5, p1[1] + 10)
        pygame.draw.polygon(self.screen, self.COLOR_ARROW, [p1, p2, p3])


    def _draw_push_indicator(self, center, direction):
        dx, dy = direction
        length = 25
        p1 = center
        p2 = (center[0] + dx * length, center[1] + dy * length)
        pygame.draw.line(self.screen, self.COLOR_ARROW, p1, p2, 3)
        # Arrowhead
        angle = math.atan2(-dy, dx)
        p3 = (p2[0] - 10 * math.cos(angle - math.pi/6), p2[1] + 10 * math.sin(angle - math.pi/6))
        p4 = (p2[0] - 10 * math.cos(angle + math.pi/6), p2[1] + 10 * math.sin(angle + math.pi/6))
        pygame.draw.polygon(self.screen, self.COLOR_ARROW, [p2, p3, p4])


    def _render_ui(self):
        # Moves counter
        moves_text = self.font_small.render(f"Moves: {self.moves} / {self.MAX_MOVES}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        # Game Over message
        if self.game_over:
            won = len(self.boxes_on_target_before_step) == len(self.box_positions)
            message = "YOU WIN!" if won else "OUT OF MOVES"
            color = self.COLOR_TARGET if won else self.COLOR_BOX
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves": self.moves,
            "boxes_on_target": len(self.boxes_on_target_before_step),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Platform Rotator")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action_taken_this_frame = False
        current_action = [0, 0, 0] # movement, space, shift
        
        # For this turn-based game, we register one action per key press.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: current_action[0] = 1
                elif event.key == pygame.K_DOWN: current_action[0] = 2
                elif event.key == pygame.K_LEFT: current_action[0] = 3
                elif event.key == pygame.K_RIGHT: current_action[0] = 4
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: current_action[2] = 1
                
                if any(a != 0 for a in current_action):
                    action_taken_this_frame = True

                if event.key == pygame.K_r: # Add a reset key for convenience
                    obs, info = env.reset()
                    done = False
                    action_taken_this_frame = False # Don't step after reset
        
        if not done and action_taken_this_frame:
            obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated
            print(f"Action: {current_action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()