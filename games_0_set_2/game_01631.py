
# Generated: 2025-08-28T02:12:28.022265
# Source Brief: brief_01631.md
# Brief Index: 1631

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to select part/location. Space to confirm. Shift to cancel part selection."

    # Must be a short, user-facing description of the game:
    game_description = "Repair a robot by placing the correct parts on damaged areas. You have limited tries!"

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.INITIAL_TRIES = 3

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (44, 48, 60)
        self.COLOR_ROBOT = (112, 128, 144)
        self.COLOR_ROBOT_OUTLINE = (80, 90, 100)
        self.COLOR_DAMAGE = (255, 190, 0)
        self.COLOR_REPAIRED = (70, 180, 120)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SUCCESS = (60, 255, 150)
        self.COLOR_FAILURE = (255, 80, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # Part definitions
        self.PART_TYPES = {
            "gear": {"color": (0, 150, 255), "shape": "circle"},
            "circuit": {"color": (0, 255, 200), "shape": "square"},
            "lens": {"color": (255, 0, 200), "shape": "diamond"},
        }
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.tries_remaining = 0
        self.game_phase = "SELECT_PART"
        self.robot_config = {}
        self.repaired_status = []
        self.available_parts = []
        self.held_part_key = None
        self.part_cursor_idx = 0
        self.location_cursor_idx = 0
        self.particles = []
        self.action_feedback = None

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def _generate_robot_config(self):
        configs = [
            {
                "grid_size": (4, 4),
                "damages": [
                    {"pos": (1, 1), "part": "gear"},
                    {"pos": (2, 3), "part": "circuit"},
                ],
                "parts_pool": ["gear", "circuit", "lens", "gear"],
            },
            {
                "grid_size": (5, 5),
                "damages": [
                    {"pos": (0, 2), "part": "lens"},
                    {"pos": (3, 1), "part": "gear"},
                    {"pos": (4, 4), "part": "circuit"},
                ],
                "parts_pool": ["gear", "circuit", "lens", "gear", "lens"],
            },
            {
                "grid_size": (3, 3),
                "damages": [
                    {"pos": (1, 2), "part": "gear"},
                ],
                "parts_pool": ["gear", "circuit", "lens"],
            },
        ]
        return self.np_random.choice(configs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.tries_remaining = self.INITIAL_TRIES
        
        self.robot_config = self._generate_robot_config()
        self.repaired_status = [False] * len(self.robot_config["damages"])
        self.available_parts = self.robot_config["parts_pool"].copy()
        
        self.game_phase = "SELECT_PART"
        self.held_part_key = None
        self.part_cursor_idx = 0
        self.location_cursor_idx = 0
        
        self.particles = []
        self.action_feedback = None
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_press = action[1] == 1  # Boolean
        shift_press = action[2] == 1  # Boolean
        
        reward = 0
        self.steps += 1
        self.action_feedback = None

        if self.game_phase == "SELECT_PART":
            if len(self.available_parts) > 0:
                if movement == 3:  # Left
                    self.part_cursor_idx = (self.part_cursor_idx - 1) % len(self.available_parts)
                elif movement == 4:  # Right
                    self.part_cursor_idx = (self.part_cursor_idx + 1) % len(self.available_parts)
            
            if space_press and self.available_parts:
                self.held_part_key = self.available_parts[self.part_cursor_idx]
                self.game_phase = "SELECT_LOCATION"
                self.location_cursor_idx = 0
                # Sound: "part_select.wav"

        elif self.game_phase == "SELECT_LOCATION":
            num_locations = len(self.robot_config["damages"])
            if movement in [1, 3]:  # Up or Left
                self.location_cursor_idx = (self.location_cursor_idx - 1) % num_locations
            elif movement in [2, 4]:  # Down or Right
                self.location_cursor_idx = (self.location_cursor_idx + 1) % num_locations

            if shift_press:  # Cancel part selection
                self.held_part_key = None
                self.game_phase = "SELECT_PART"
                # Sound: "cancel.wav"
            elif space_press:
                target_idx = self.location_cursor_idx
                
                if not self.repaired_status[target_idx]:
                    correct_part = self.robot_config["damages"][target_idx]["part"]
                    
                    if self.held_part_key == correct_part:
                        reward += 10
                        self.score += 10
                        self.repaired_status[target_idx] = True
                        self.action_feedback = {"type": "success", "idx": target_idx}
                        
                        self.available_parts.pop(self.part_cursor_idx)
                        if self.part_cursor_idx >= len(self.available_parts) and self.available_parts:
                            self.part_cursor_idx = len(self.available_parts) - 1
                        
                        if all(self.repaired_status):
                            self.win_condition = True
                            self.game_over = True
                            bonus = self.tries_remaining * 10
                            reward += 100 + bonus
                            self.score += 100 + bonus
                            # Sound: "win_game.wav"
                        else:
                            # Sound: "correct_placement.wav"
                            pass
                    else:
                        reward -= 10
                        self.score -= 10
                        self.tries_remaining -= 1
                        self.action_feedback = {"type": "failure", "idx": target_idx}
                        # Sound: "incorrect_placement.wav"
                        
                        if self.tries_remaining <= 0:
                            self.game_over = True
                            # Sound: "lose_game.wav"
                
                self.held_part_key = None
                self.game_phase = "SELECT_PART"
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._draw_grid()
        self._draw_robot()
        self._draw_parts_ui()
        self._update_and_draw_particles()
        
        # Render UI overlay
        self._draw_ui_overlay()

        if self.game_over:
            self._draw_game_over_screen()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tries_remaining": self.tries_remaining,
        }

    def _iso_to_screen(self, x, y, tile_w=32, tile_h=16):
        screen_x = self.WIDTH // 2 + (x - y) * tile_w / 2
        screen_y = self.HEIGHT // 3 + (x + y) * tile_h / 2
        return int(screen_x), int(screen_y)

    def _draw_grid(self):
        grid_w, grid_h = self.robot_config["grid_size"]
        for x in range(grid_w + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, grid_h)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for y in range(grid_h + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(grid_w, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _draw_iso_poly(self, surface, points, color, outline_color=None):
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _draw_robot(self):
        grid_w, grid_h = self.robot_config["grid_size"]
        tile_w, tile_h = 32, 16
        
        p1 = self._iso_to_screen(0, 0)
        p2 = self._iso_to_screen(grid_w, 0)
        p3 = self._iso_to_screen(grid_w, grid_h)
        p4 = self._iso_to_screen(0, grid_h)
        self._draw_iso_poly(self.screen, [p1, p2, p3, p4], self.COLOR_ROBOT, self.COLOR_ROBOT_OUTLINE)

        for i, damage in enumerate(self.robot_config["damages"]):
            gx, gy = damage["pos"]
            center_x, center_y = self._iso_to_screen(gx, gy)
            center_y += tile_h // 2
            points = [
                (center_x, center_y - tile_h // 2), (center_x + tile_w // 2, center_y),
                (center_x, center_y + tile_h // 2), (center_x - tile_w // 2, center_y)
            ]
            color = self.COLOR_REPAIRED if self.repaired_status[i] else self.COLOR_DAMAGE
            self._draw_iso_poly(self.screen, points, color)
            
            if self.game_phase == "SELECT_LOCATION" and self.location_cursor_idx == i:
                pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
                cursor_color = tuple(int(c * pulse + self.COLOR_CURSOR[j] * (1 - pulse)) for j, c in enumerate((200, 200, 255)))
                pygame.gfxdraw.aapolygon(self.screen, points, cursor_color)
                pygame.gfxdraw.aapolygon(self.screen, [(p[0] + 1, p[1]) for p in points], cursor_color)

        if self.action_feedback:
            idx = self.action_feedback["idx"]
            gx, gy = self.robot_config["damages"][idx]["pos"]
            center_x, center_y = self._iso_to_screen(gx, gy)
            center_y += tile_h // 2
            if self.action_feedback["type"] == "success":
                self._create_particles(center_x, center_y, self.COLOR_SUCCESS, 20)
            elif self.action_feedback["type"] == "failure":
                self._create_particles(center_x, center_y, self.COLOR_FAILURE, 10, speed=2)
            self.action_feedback = None

    def _draw_part(self, surface, pos, part_key, size=20):
        part_info = self.PART_TYPES[part_key]
        x, y = pos
        color = part_info["color"]
        if part_info["shape"] == "circle":
            pygame.gfxdraw.aacircle(surface, x, y, size // 2, color)
            pygame.gfxdraw.filled_circle(surface, x, y, size // 2, color)
        elif part_info["shape"] == "square":
            rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
            pygame.draw.rect(surface, color, rect)
        elif part_info["shape"] == "diamond":
            points = [(x, y - size // 2), (x + size // 2, y), (x, y + size // 2), (x - size // 2, y)]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_parts_ui(self):
        panel_y = self.HEIGHT - 50
        total_width = len(self.available_parts) * 60
        start_x = self.WIDTH // 2 - total_width // 2

        for i, part_key in enumerate(self.available_parts):
            x = start_x + i * 60 + 30
            if self.game_phase == "SELECT_PART" and i == self.part_cursor_idx:
                pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 * 10 + 5
                pygame.gfxdraw.aacircle(self.screen, x, panel_y, 20 + int(pulse), self.COLOR_CURSOR)
            
            pygame.gfxdraw.box(self.screen, pygame.Rect(x - 25, panel_y - 25, 50, 50), (*self.COLOR_GRID, 150))
            self._draw_part(self.screen, (x, panel_y), part_key)

        if self.held_part_key:
            target_idx = self.location_cursor_idx
            gx, gy = self.robot_config["damages"][target_idx]["pos"]
            screen_x, screen_y = self._iso_to_screen(gx, gy)
            self._draw_part(self.screen, (screen_x + 40, screen_y - 20), self.held_part_key, size=25)

    def _draw_text(self, text, pos, font, color, shadow_color=None, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center: text_rect.center = pos
        else: text_rect.topleft = pos
        if shadow_color:
            shadow_surf = font.render(text, True, shadow_color)
            shadow_rect = shadow_surf.get_rect(topleft=(text_rect.left + 2, text_rect.top + 2))
            if center: shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _draw_ui_overlay(self):
        self._draw_text(f"SCORE: {self.score}", (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        tries_text = "TRIES: " + "● " * self.tries_remaining + "○ " * (self.INITIAL_TRIES - self.tries_remaining)
        self._draw_text(tries_text, (self.WIDTH - 150, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        repaired, total = sum(self.repaired_status), len(self.repaired_status)
        self._draw_text(f"REPAIRED: {repaired}/{total}", (10, 40), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _draw_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        message = "REPAIR COMPLETE" if self.win_condition else "SYSTEM FAILURE"
        color = self.COLOR_SUCCESS if self.win_condition else self.COLOR_FAILURE
        self._draw_text(message, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_big, color, self.COLOR_TEXT_SHADOW, center=True)
        self._draw_text(f"Final Score: {self.score}", (self.WIDTH // 2, self.HEIGHT // 2 + 40), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)

    def _create_particles(self, x, y, color, count, speed=4):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_x = math.cos(angle) * self.np_random.uniform(0.5, speed)
            vel_y = math.sin(angle) * self.np_random.uniform(0.5, speed)
            lifetime = self.np_random.integers(20, 40)
            self.particles.append([[x, y], [vel_x, vel_y], self.np_random.integers(3, 7), color, lifetime])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[4] -= 1
            p[2] -= 0.1
            if p[4] > 0 and p[2] > 0:
                pos, size = (int(p[0][0]), int(p[0][1])), int(p[2])
                alpha = max(0, min(255, int(255 * (p[4] / 40))))
                color = (*p[3], alpha)
                temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, size, size, size, color)
                self.screen.blit(temp_surf, (pos[0] - size, pos[1] - size))
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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