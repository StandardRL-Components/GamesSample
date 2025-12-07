
# Generated: 2025-08-28T05:30:36.444931
# Source Brief: brief_02651.md
# Brief Index: 2651

        
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
    An isometric puzzle game where the player must push colored boxes onto their
    corresponding targets within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move and push boxes. "
        "The goal is to place all boxes on their matching colored targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A Sokoban-style isometric puzzle. Push boxes to their designated locations in a warehouse "
        "within a limited number of moves. Careful planning is key!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 8
        self.TILE_WIDTH_ISO = 64
        self.TILE_HEIGHT_ISO = 32
        self.TILE_DEPTH_ISO = 24

        # Center the grid on the screen
        self.OFFSET_X = self.SCREEN_WIDTH // 2
        self.OFFSET_Y = self.SCREEN_HEIGHT // 2 - self.GRID_HEIGHT * self.TILE_HEIGHT_ISO // 2 + 20

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_WALL = (80, 85, 90)
        self.COLOR_WALL_TOP = (100, 105, 110)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BOX_COLORS = {
            "red": ((220, 50, 50), (180, 40, 40), (140, 30, 30)),
            "blue": ((50, 120, 220), (40, 100, 180), (30, 80, 140)),
            "green": ((50, 220, 120), (40, 180, 100), (30, 140, 80)),
            "yellow": ((220, 200, 50), (180, 160, 40), (140, 120, 30)),
        }
        self.TARGET_COLORS = {
            "red": (100, 30, 30),
            "blue": (30, 60, 100),
            "green": (30, 100, 60),
            "yellow": (100, 90, 30),
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables
        self.player_pos = None
        self.boxes = None
        self.targets = None
        self.walls = None
        self.moves_remaining = None
        self.last_move_direction = None
        self.rewards_collected_for_box = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False

        self.reset()
        self.validate_implementation()

    def _define_level(self):
        """Defines the layout of the puzzle."""
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, -1))
            self.walls.add((x, self.GRID_HEIGHT))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((-1, y))
            self.walls.add((self.GRID_WIDTH, y))

        # Border walls
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(1, self.GRID_HEIGHT - 1):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))
        
        # Internal walls
        self.walls.update([(3,3), (4,3), (5,3), (3,4), (5,4)])

        self.player_pos = (1, 1)
        
        self.boxes = [
            {"pos": (2, 2), "color": "red"},
            {"pos": (4, 2), "color": "blue"},
            {"pos": (2, 5), "color": "green"},
            {"pos": (6, 5), "color": "yellow"},
        ]
        
        self.targets = [
            {"pos": (7, 2), "color": "red"},
            {"pos": (7, 4), "color": "blue"},
            {"pos": (1, 6), "color": "green"},
            {"pos": (4, 6), "color": "yellow"},
        ]
        
        self.rewards_collected_for_box = [False] * len(self.boxes)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._define_level()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_remaining = 20
        self.last_move_direction = (0, 1)  # Down

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0
        terminated = False
        
        # Only process non-noop actions
        if movement > 0:
            self.steps += 1
            self.moves_remaining -= 1
            reward -= 0.1 # Cost for making a move

            # Map movement to delta
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
            dx, dy = move_map[movement]
            self.last_move_direction = (dx, dy)

            player_x, player_y = self.player_pos
            next_x, next_y = player_x + dx, player_y + dy

            # Check for wall collision
            if (next_x, next_y) in self.walls:
                # Player bumps into wall, no state change
                # sound_effect: 'bump_wall'
                pass
            else:
                # Check for box at next position
                box_idx = self._get_box_at(next_x, next_y)
                if box_idx is not None:
                    # Attempt to push the box
                    box_next_x, box_next_y = next_x + dx, next_y + dy
                    
                    is_obstructed = (box_next_x, box_next_y) in self.walls or \
                                    self._get_box_at(box_next_x, box_next_y) is not None

                    if not is_obstructed:
                        # Push succeeds
                        self.boxes[box_idx]["pos"] = (box_next_x, box_next_y)
                        self.player_pos = (next_x, next_y)
                        # sound_effect: 'push_box'
                    else:
                        # Push fails
                        # sound_effect: 'bump_box'
                        pass
                else:
                    # Simple movement
                    self.player_pos = (next_x, next_y)
                    # sound_effect: 'step'

        # Calculate rewards and check for win condition
        all_boxes_on_target = True
        for i, box in enumerate(self.boxes):
            is_on_target = any(
                box["pos"] == target["pos"] and box["color"] == target["color"]
                for target in self.targets
            )

            if is_on_target:
                if not self.rewards_collected_for_box[i]:
                    reward += 1.0
                    self.rewards_collected_for_box[i] = True
            else:
                # If box was moved off its target, reset reward flag
                if self.rewards_collected_for_box[i]:
                    self.rewards_collected_for_box[i] = False
                all_boxes_on_target = False

        if all_boxes_on_target:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.win_state = True
        
        if self.moves_remaining <= 0 and not self.win_state:
            reward -= 100.0
            terminated = True
            self.game_over = True
            self.win_state = False

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_box_at(self, x, y):
        for i, box in enumerate(self.boxes):
            if box["pos"] == (x, y):
                return i
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = (grid_x - grid_y) * self.TILE_WIDTH_ISO / 2
        screen_y = (grid_x + grid_y) * self.TILE_HEIGHT_ISO / 2
        return int(self.OFFSET_X + screen_x), int(self.OFFSET_Y + screen_y)

    def _draw_iso_poly(self, points, color, outline_color=None, width=0):
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)

    def _draw_iso_cube(self, pos, color_tuple, height_mod=1.0):
        x, y = pos
        top_color, side_dark_color, side_light_color = color_tuple
        
        tile_depth = self.TILE_DEPTH_ISO * height_mod

        p_top_left = self._iso_to_screen(x, y)
        p_top_right = self._iso_to_screen(x + 1, y)
        p_bottom_left = self._iso_to_screen(x, y + 1)
        p_bottom_right = self._iso_to_screen(x + 1, y + 1)

        # Top face
        top_points = [
            (p_top_left[0], p_top_left[1] - tile_depth),
            (p_top_right[0], p_top_right[1] - tile_depth),
            (p_bottom_right[0], p_bottom_right[1] - tile_depth),
            (p_bottom_left[0], p_bottom_left[1] - tile_depth),
        ]
        self._draw_iso_poly(top_points, top_color)

        # Left face
        left_points = [
            p_bottom_left,
            p_bottom_right,
            (p_bottom_right[0], p_bottom_right[1] - tile_depth),
            (p_bottom_left[0], p_bottom_left[1] - tile_depth),
        ]
        self._draw_iso_poly(left_points, side_dark_color)

        # Right face
        right_points = [
            p_top_right,
            p_bottom_right,
            (p_bottom_right[0], p_bottom_right[1] - tile_depth),
            (p_top_right[0], p_top_right[1] - tile_depth),
        ]
        self._draw_iso_poly(right_points, side_light_color)

    def _render_game(self):
        # Draw floor grid and targets first
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                is_wall_tile = (x, y) in self.walls
                if not is_wall_tile:
                    p1 = self._iso_to_screen(x, y)
                    p2 = self._iso_to_screen(x + 1, y)
                    p3 = self._iso_to_screen(x + 1, y + 1)
                    p4 = self._iso_to_screen(x, y + 1)
                    pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_GRID)

        for target in self.targets:
            x, y = target["pos"]
            color = self.TARGET_COLORS[target["color"]]
            outline_color = self.BOX_COLORS[target["color"]][0]
            p1 = self._iso_to_screen(x, y)
            p2 = self._iso_to_screen(x + 1, y)
            p3 = self._iso_to_screen(x + 1, y + 1)
            p4 = self._iso_to_screen(x, y + 1)
            self._draw_iso_poly([p1, p2, p3, p4], color, outline_color)

        # Create a sorted list of all dynamic entities for painter's algorithm
        entities = []
        for wall_pos in self.walls:
            entities.append(("wall", wall_pos, 0))
        for i, box in enumerate(self.boxes):
            entities.append(("box", box["pos"], i))
        entities.append(("player", self.player_pos, -1))
        
        # Sort by y then x for correct occlusion
        entities.sort(key=lambda e: (e[1][0] + e[1][1], e[1][1]))

        for entity_type, pos, data_idx in entities:
            if entity_type == "wall":
                self._draw_iso_cube(pos, (self.COLOR_WALL_TOP, self.COLOR_WALL, self.COLOR_WALL), height_mod=1.5)
            elif entity_type == "box":
                box = self.boxes[data_idx]
                colors = self.BOX_COLORS[box["color"]]
                self._draw_iso_cube(box["pos"], colors)
                
                # Add a glow if on target
                is_on_target = any(box["pos"] == t["pos"] and box["color"] == t["color"] for t in self.targets)
                if is_on_target:
                    center_x, center_y = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
                    center_y -= self.TILE_DEPTH_ISO / 2
                    glow_color = (*colors[0], 60) # RGBA
                    radius = int(self.TILE_WIDTH_ISO * 0.6)
                    temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, glow_color, (radius, radius), radius)
                    self.screen.blit(temp_surf, (center_x - radius, center_y - radius), special_flags=pygame.BLEND_RGBA_ADD)

            elif entity_type == "player":
                px, py = self.player_pos
                center_x, center_y = self._iso_to_screen(px + 0.5, py + 0.5)
                center_y -= self.TILE_DEPTH_ISO / 2 # Elevate slightly
                
                dx, dy = self.last_move_direction
                angle = math.atan2(dx + dy, dx - dy)
                
                size = 12
                points = [
                    (center_x + size * math.cos(angle), center_y + size * math.sin(angle)),
                    (center_x + size * math.cos(angle + 2.5), center_y + size * math.sin(angle + 2.5)),
                    (center_x + size * 0.3 * math.cos(angle + math.pi), center_y + size * 0.3 * math.sin(angle + math.pi)),
                    (center_x + size * math.cos(angle - 2.5), center_y + size * math.sin(angle - 2.5)),
                ]
                self._draw_iso_poly(points, self.COLOR_PLAYER, (180, 180, 180))

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Game over messages
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_state:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "OUT OF MOVES"
                color = (255, 100, 100)
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Warehouse")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                
                # Only step if a move key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_remaining']}")

        # Draw the observation to the display screen
        draw_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(draw_obs)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()