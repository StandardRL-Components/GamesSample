
# Generated: 2025-08-28T02:12:00.135108
# Source Brief: brief_01629.md
# Brief Index: 1629

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper for 2D vector operations
Vec2 = namedtuple('Vec2', ['x', 'y'])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to pick up or rotate an organ. "
        "Hold Shift to place the held organ."
    )

    game_description = (
        "Perform isometric 2D autopsies on quirky aliens by rotating and positioning organs to match a "
        "target blueprint within a limited number of moves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 14
    ISO_TILE_WIDTH, ISO_TILE_HEIGHT = 32, 16
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 80

    # --- Colors ---
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (40, 45, 50)
    COLOR_CAVITY = (20, 22, 25)
    COLOR_TARGET_OUTLINE = (70, 80, 90)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_UI_SUCCESS = (100, 255, 150)
    COLOR_UI_FAIL = (255, 100, 100)
    COLOR_CURSOR = (0, 150, 255)
    COLOR_CURSOR_HIGHLIGHT = (0, 200, 255, 100)
    
    ORGAN_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
        (255, 160, 80),  # Orange
    ]

    # --- Organ Shapes (list of Vec2 points relative to center) ---
    ORGAN_SHAPES = [
        [Vec2(0, 0)],  # Single block
        [Vec2(0, 0), Vec2(1, 0)],  # I-shape (2)
        [Vec2(0, 0), Vec2(1, 0), Vec2(-1, 0)],  # I-shape (3)
        [Vec2(0, 0), Vec2(1, 0), Vec2(0, 1)],  # L-shape
        [Vec2(0, 0), Vec2(1, 0), Vec2(-1, 0), Vec2(0, 1)],  # T-shape
        [Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), Vec2(1, 1)],  # Square
        [Vec2(0, 0), Vec2(1, 0), Vec2(-1, 0), Vec2(1, 1), Vec2(-1, 1)], # U-shape
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_score = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 48)

        self.organs = []
        self.particles = []
        self.cursor_pos = Vec2(0, 0)
        self.held_organ_idx = None
        self.autopsies_completed = 0
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_feedback_msg = ""
        self.last_feedback_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.autopsies_completed = 0
        self.last_feedback_msg = ""
        self.last_feedback_timer = 0
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.organs.clear()
        self.particles.clear()
        self.held_organ_idx = None
        
        num_organs = min(3 + self.autopsies_completed, len(self.ORGAN_SHAPES))
        self.moves_remaining = 50 + 10 * self.autopsies_completed
        self.cursor_pos = Vec2(self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2)

        # Generate valid target positions within the cavity
        cavity_positions = []
        for y in range(self.GRID_HEIGHT - 6):
            for x in range(self.GRID_WIDTH):
                if 2 < x < self.GRID_WIDTH - 3 and 1 < y < self.GRID_HEIGHT - 7:
                    cavity_positions.append(Vec2(x,y))
        
        self.np_random.shuffle(cavity_positions)
        
        # Generate organs
        used_shapes = list(range(len(self.ORGAN_SHAPES)))
        self.np_random.shuffle(used_shapes)
        
        for i in range(num_organs):
            shape_idx = used_shapes[i % len(used_shapes)]
            shape = self.ORGAN_SHAPES[shape_idx]
            color = self.ORGAN_COLORS[i % len(self.ORGAN_COLORS)]
            
            target_pos = cavity_positions.pop(0)
            target_rot = self.np_random.integers(0, 4)
            
            initial_pos = Vec2(
                self.np_random.integers(2, self.GRID_WIDTH - 2),
                self.np_random.integers(self.GRID_HEIGHT - 4, self.GRID_HEIGHT - 1)
            )

            self.organs.append({
                "id": i,
                "shape": shape,
                "color": color,
                "pos": initial_pos,
                "rotation": self.np_random.integers(0, 4),
                "target_pos": target_pos,
                "target_rotation": target_rot,
                "is_placed": False,
                "dist_to_target": self._dist(initial_pos, target_pos)
            })
        self.selected_organ_idx = None

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        is_meaningful_action = any(a != 0 for a in action)
        if is_meaningful_action:
            self.moves_remaining -= 1
        
        # 1. Handle Movement
        if movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_cursor_pos = Vec2(
                max(0, min(self.GRID_WIDTH - 1, self.cursor_pos.x + dx)),
                max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos.y + dy))
            )
            self.cursor_pos = new_cursor_pos

            if self.held_organ_idx is not None:
                organ = self.organs[self.held_organ_idx]
                organ["pos"] = self.cursor_pos
                
                new_dist = self._dist(organ["pos"], organ["target_pos"])
                if new_dist < organ["dist_to_target"]:
                    reward += 1.0  # Closer to target
                elif new_dist > organ["dist_to_target"]:
                    reward -= 0.1  # Further from target
                organ["dist_to_target"] = new_dist

        # 2. Handle Space (Pick up / Rotate)
        if space_press:
            if self.held_organ_idx is not None:
                # Rotate held organ
                organ = self.organs[self.held_organ_idx]
                organ["rotation"] = (organ["rotation"] + 1) % 4
                # sound: "organ_rotate.wav"
            else:
                # Try to pick up an organ
                for i, organ in enumerate(self.organs):
                    if not organ["is_placed"] and organ["pos"] == self.cursor_pos:
                        self.held_organ_idx = i
                        # sound: "organ_pickup.wav"
                        break
        
        # 3. Handle Shift (Place)
        if shift_press and self.held_organ_idx is not None:
            organ = self.organs[self.held_organ_idx]
            
            is_correct_pos = (organ["pos"] == organ["target_pos"])
            is_correct_rot = (organ["rotation"] == organ["target_rotation"])

            if is_correct_pos and is_correct_rot:
                organ["is_placed"] = True
                reward += 10.0
                self._show_feedback("Correct Placement!", self.COLOR_UI_SUCCESS)
                self._create_particles(organ["pos"], organ["color"])
                # sound: "correct_placement.wav"
            else:
                self._show_feedback("Incorrect!", self.COLOR_UI_FAIL)
                # sound: "incorrect_placement.wav"

            self.held_organ_idx = None

        # 4. Check for level completion
        if all(o["is_placed"] for o in self.organs):
            self.autopsies_completed += 1
            reward += 100.0
            # sound: "level_complete.wav"
            
            if self.autopsies_completed < 5:
                self._show_feedback(f"Autopsy {self.autopsies_completed} Complete!", self.COLOR_UI_SUCCESS)
                self._setup_level()
            else:
                self.game_over = True # Win
                self._show_feedback("All Autopsies Complete!", self.COLOR_UI_SUCCESS)

        # 5. Check for termination
        if self.moves_remaining <= 0 and not self.game_over:
            self.game_over = True # Lose
            reward -= 100.0
            self._show_feedback("Out of Moves!", self.COLOR_UI_FAIL)
            # sound: "game_over.wav"
        
        if self.steps >= 1000:
            self.game_over = True

        self.steps += 1
        self.score += reward
        
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "autopsies_completed": self.autopsies_completed,
            "moves_remaining": self.moves_remaining,
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.ISO_TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.ISO_TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _rotate_point(self, point, rotation):
        rad = math.radians(90 * rotation)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # Use integer-based rotation for perfect 90-degree turns
        if rotation == 0: return point
        if rotation == 1: return Vec2(-point.y, point.x)
        if rotation == 2: return Vec2(-point.x, -point.y)
        if rotation == 3: return Vec2(point.y, -point.x)
        return point

    def _draw_polygon_alpha(self, surface, color, points):
        lx, ly = zip(*points)
        min_x, max_x = min(lx), max(lx)
        min_y, max_y = min(ly), max(ly)
        target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        
        adj_points = [(x - min_x, y - min_y) for x, y in points]
        pygame.draw.polygon(shape_surf, color, adj_points)
        surface.blit(shape_surf, target_rect)

    def _render_game(self):
        # Draw grid and cavity
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.gfxdraw.line(self.screen, p1[0], p1[1], p2[0], p2[1], self.COLOR_GRID)
                pygame.gfxdraw.line(self.screen, p2[0], p2[1], p3[0], p3[1], self.COLOR_GRID)

        cavity_points = [
            self._iso_to_screen(3, 2), self._iso_to_screen(self.GRID_WIDTH-3, 2),
            self._iso_to_screen(self.GRID_WIDTH-3, self.GRID_HEIGHT-7),
            self._iso_to_screen(3, self.GRID_HEIGHT-7)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, cavity_points, self.COLOR_CAVITY)
        pygame.gfxdraw.aapolygon(self.screen, cavity_points, self.COLOR_TARGET_OUTLINE)

        # Create a list of all items to draw for Z-sorting
        draw_list = []

        # Add targets for unplaced organs
        for organ in self.organs:
            if not organ["is_placed"]:
                draw_list.append(("target", organ))
        
        # Add organs
        for i, organ in enumerate(self.organs):
             if self.held_organ_idx != i:
                draw_list.append(("organ", organ))

        # Sort by grid y-pos for correct isometric rendering
        draw_list.sort(key=lambda item: item[1]["pos"].y)
        
        # Draw sorted items
        for item_type, organ in draw_list:
            self._render_organ(organ, is_target=(item_type == "target"))
            
        # Draw held organ last so it's on top
        if self.held_organ_idx is not None:
            self._render_organ(self.organs[self.held_organ_idx], is_held=True)
        
        # Draw cursor
        cursor_screen_pos = self._iso_to_screen(self.cursor_pos.x, self.cursor_pos.y)
        s = self.ISO_TILE_WIDTH / 2
        cursor_poly = [
            (cursor_screen_pos[0], cursor_screen_pos[1] - s/2),
            (cursor_screen_pos[0] + s, cursor_screen_pos[1]),
            (cursor_screen_pos[0], cursor_screen_pos[1] + s/2),
            (cursor_screen_pos[0] - s, cursor_screen_pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, cursor_poly, self.COLOR_CURSOR)
        if self.held_organ_idx is None:
             pygame.gfxdraw.filled_polygon(self.screen, cursor_poly, self.COLOR_CURSOR_HIGHLIGHT)

        # Update and draw particles
        self._update_particles()

    def _render_organ(self, organ, is_target=False, is_held=False):
        pos = organ["target_pos"] if is_target else organ["pos"]
        rot = organ["target_rotation"] if is_target else organ["rotation"]
        
        for block in organ["shape"]:
            rotated_block = self._rotate_point(block, rot)
            block_pos = Vec2(pos.x + rotated_block.x, pos.y + rotated_block.y)
            
            p1 = self._iso_to_screen(block_pos.x, block_pos.y)
            p2 = self._iso_to_screen(block_pos.x + 1, block_pos.y)
            p3 = self._iso_to_screen(block_pos.x + 1, block_pos.y + 1)
            p4 = self._iso_to_screen(block_pos.x, block_pos.y + 1)
            points = [p1, p2, p3, p4]

            if is_target:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TARGET_OUTLINE)
            else:
                color = organ["color"]
                if organ["is_placed"]:
                    # Slightly desaturated when placed
                    color = tuple(max(0, c-40) for c in color)
                
                top_color = tuple(min(255, c+40) for c in color)
                side_color = tuple(max(0, c-40) for c in color)

                # Draw sides
                pygame.gfxdraw.filled_polygon(self.screen, [p4, p3, (p3[0], p3[1]+8), (p4[0], p4[1]+8)], side_color)
                pygame.gfxdraw.filled_polygon(self.screen, [p2, p3, (p3[0], p3[1]+8), (p2[0], p2[1]+8)], color)
                # Draw top
                pygame.gfxdraw.filled_polygon(self.screen, points, top_color)
                # Draw outline
                pygame.gfxdraw.aapolygon(self.screen, points, (0,0,0))
                
                if is_held:
                    highlight_color = (*self.COLOR_CURSOR, 100)
                    self._draw_polygon_alpha(self.screen, highlight_color, points)

    def _render_ui(self):
        # Autopsies completed
        text = self.font_ui.render("Autopsies:", True, self.COLOR_UI_TEXT)
        self.screen.blit(text, (10, 10))
        value = self.font_score.render(f"{self.autopsies_completed} / 5", True, self.COLOR_UI_VALUE)
        self.screen.blit(value, (10, 30))
        
        # Moves remaining
        text = self.font_ui.render("Moves Left:", True, self.COLOR_UI_TEXT)
        text_rect = text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text, text_rect)
        moves_color = self.COLOR_UI_VALUE if self.moves_remaining > 10 else self.COLOR_UI_FAIL
        value = self.font_score.render(f"{self.moves_remaining}", True, moves_color)
        value_rect = value.get_rect(topright=(self.SCREEN_WIDTH - 10, 30))
        self.screen.blit(value, value_rect)

        # Score
        text = self.font_ui.render("Score", True, self.COLOR_UI_TEXT)
        text_rect = text.get_rect(midbottom=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30))
        self.screen.blit(text, text_rect)
        value = self.font_score.render(f"{int(self.score)}", True, self.COLOR_UI_VALUE)
        value_rect = value.get_rect(midbottom=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 5))
        self.screen.blit(value, value_rect)

        # Feedback message
        if self.last_feedback_timer > 0:
            msg_surf = self.font_msg.render(self.last_feedback_msg, True, self.last_feedback_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)
            self.last_feedback_timer -= 1

    def _show_feedback(self, msg, color):
        self.last_feedback_msg = msg
        self.last_feedback_color = color
        self.last_feedback_timer = 30 # Show for 30 frames (since auto_advance=False, this is 30 steps)

    def _dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _create_particles(self, grid_pos, color):
        screen_pos = self._iso_to_screen(grid_pos.x + 0.5, grid_pos.y + 0.5)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = Vec2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": list(screen_pos),
                "vel": vel,
                "life": self.np_random.integers(20, 40),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"].x
            p["pos"][1] += p["vel"].y
            p["life"] -= 1
            if p["life"] > 0:
                alpha = max(0, min(255, int(255 * (p["life"] / 30))))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"][0]), int(p["pos"][1]), 
                    max(1, int(p["life"]/8)), (*p["color"], alpha)
                )
        self.particles = [p for p in self.particles if p["life"] > 0]
    
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Alien Autopsy")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n--- Manual Control ---")
    print(env.user_guide)
    print("----------------------")

    while not terminated:
        action = [0, 0, 0] # no-op, movement, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_remaining']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit manual play speed
        
        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            pygame.time.wait(3000)

    env.close()