
# Generated: 2025-08-27T18:56:47.008166
# Source Brief: brief_01994.md
# Brief Index: 1994

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Stand on a bush and press Space to hide. Avoid the guards!"
    )

    game_description = (
        "A stealthy squirrel adventure! Navigate the garden, hide in bushes, and snatch the golden acorn before the guards catch you or time runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.TILE_SIZE = 32
        self.UI_HEIGHT = self.SCREEN_HEIGHT - (self.GRID_HEIGHT * self.TILE_SIZE) # 16px

        self.MAX_STEPS = 30
        self.MAX_DETECTIONS = 2

        # --- Colors ---
        self.COLOR_BG_DARK = (45, 65, 45)
        self.COLOR_BG_LIGHT = (55, 75, 55)
        self.COLOR_SQUIRREL = (160, 82, 45)
        self.COLOR_SQUIRREL_OUTLINE = (90, 40, 20)
        self.COLOR_GUARD = (100, 110, 120)
        self.COLOR_GUARD_OUTLINE = (60, 70, 80)
        self.COLOR_BUSH = (34, 139, 34)
        self.COLOR_BUSH_OUTLINE = (20, 80, 20)
        self.COLOR_ACORN = (255, 215, 0)
        self.COLOR_ACORN_STEM = (139, 69, 19)
        self.COLOR_VISION_CONE = (255, 255, 0, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        self.COLOR_DETECTION = (255, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_detection = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_ui = pygame.font.SysFont("arial", 18)
            self.font_detection = pygame.font.SysFont("arial", 48, bold=True)

        # --- Game State Initialization ---
        self.squirrel_pos = None
        self.is_hidden = None
        self.guards = None
        self.bushes = None
        self.acorn_pos = None
        self.steps = None
        self.score = None
        self.detections = None
        self.detection_effects = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.detections = 0
        self.is_hidden = False
        self.detection_effects = []

        # --- Level Layout (Fixed for consistency) ---
        self.squirrel_pos = np.array([1, self.GRID_HEIGHT // 2])
        self.acorn_pos = np.array([self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2])
        
        self.bushes = [
            np.array([4, 2]), np.array([5, 2]),
            np.array([4, 9]), np.array([5, 9]),
            np.array([9, 5]), np.array([9, 6]), np.array([10, 5]), np.array([10, 6]),
            np.array([14, 2]), np.array([15, 2]),
            np.array([14, 9]), np.array([15, 9]),
        ]

        # --- Guard Initialization ---
        self.guards = [
            {
                "path": [np.array([7, 1]), np.array([7, self.GRID_HEIGHT - 2])],
                "pos": np.array([7, 1], dtype=np.float64),
                "dir": np.array([0, 1], dtype=np.float64),
                "target_idx": 1,
                "speed": 0.1,
                "vision_range": 5,
                "vision_angle": math.pi / 4,
            },
            {
                "path": [np.array([12, self.GRID_HEIGHT - 2]), np.array([12, 1])],
                "pos": np.array([12, self.GRID_HEIGHT - 2], dtype=np.float64),
                "dir": np.array([0, -1], dtype=np.float64),
                "target_idx": 1,
                "speed": 0.1,
                "vision_range": 5,
                "vision_angle": math.pi / 4,
            },
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = -0.1  # Cost for taking a step
        
        # --- 1. Update Player State ---
        old_pos = self.squirrel_pos.copy()
        
        if movement == 1:  # Up
            self.squirrel_pos[1] -= 1
        elif movement == 2:  # Down
            self.squirrel_pos[1] += 1
        elif movement == 3:  # Left
            self.squirrel_pos[0] -= 1
        elif movement == 4:  # Right
            self.squirrel_pos[0] += 1

        self.squirrel_pos[0] = np.clip(self.squirrel_pos[0], 0, self.GRID_WIDTH - 1)
        self.squirrel_pos[1] = np.clip(self.squirrel_pos[1], 0, self.GRID_HEIGHT - 1)
        
        moved = not np.array_equal(old_pos, self.squirrel_pos)
        
        # Handle hiding
        on_bush = any(np.array_equal(self.squirrel_pos, b) for b in self.bushes)
        self.is_hidden = False
        if space_held == 1 and on_bush:
            self.is_hidden = True
            if not moved: # Only reward for intentionally hiding, not just passing through
                 reward += 0.5 # Reward for successfully hiding

        # --- 2. Update Guard State & Check Detection ---
        detected_this_step = False
        for guard in self.guards:
            # Move guard along path
            target_pos = guard["path"][guard["target_idx"]]
            direction_to_target = target_pos - guard["pos"]
            distance = np.linalg.norm(direction_to_target)

            if distance < guard["speed"]:
                guard["pos"] = target_pos.astype(np.float64)
                guard["target_idx"] = (guard["target_idx"] + 1) % len(guard["path"])
                new_target_pos = guard["path"][guard["target_idx"]]
                guard["dir"] = (new_target_pos - guard["pos"]) / np.linalg.norm(new_target_pos - guard["pos"])
            else:
                guard["dir"] = direction_to_target / distance
                guard["pos"] += guard["dir"] * guard["speed"]

            # Check detection
            if not self.is_hidden and self._is_in_vision(guard, self.squirrel_pos):
                if not detected_this_step:
                    self.detections += 1
                    reward -= 5.0
                    # sfx: !
                    self.detection_effects.append({"pos": guard["pos"].copy(), "timer": 15})
                    detected_this_step = True
        
        # Update detection effects timer
        self.detection_effects = [e for e in self.detection_effects if e["timer"] > 0]
        for effect in self.detection_effects:
            effect["timer"] -= 1

        # --- 3. Check Termination Conditions ---
        self.steps += 1
        terminated = False
        
        if np.array_equal(self.squirrel_pos, self.acorn_pos):
            reward += 100.0
            terminated = True
            # sfx: win_sound
        elif self.detections >= self.MAX_DETECTIONS:
            reward -= 100.0
            terminated = True
            # sfx: lose_sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _is_in_vision(self, guard, target_pos):
        guard_pos = guard["pos"]
        vec_to_target = target_pos - guard_pos
        dist_to_target = np.linalg.norm(vec_to_target)

        if dist_to_target == 0 or dist_to_target > guard["vision_range"]:
            return False

        vec_to_target_norm = vec_to_target / dist_to_target
        angle_between = math.acos(np.clip(np.dot(vec_to_target_norm, guard["dir"]), -1.0, 1.0))

        return angle_between <= guard["vision_angle"]

    def _get_observation(self):
        self._render_background()
        self._render_level_objects()
        self._render_guards()
        self._render_player()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "detections": self.detections,
            "squirrel_pos": self.squirrel_pos.tolist(),
            "is_hidden": self.is_hidden,
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_DARK)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x + y) % 2 == 0:
                    pygame.draw.rect(
                        self.screen,
                        self.COLOR_BG_LIGHT,
                        (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE),
                    )
    
    def _render_level_objects(self):
        # Draw Bushes
        for pos in self.bushes:
            cx = int(pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
            cy = int(pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
            radius = int(self.TILE_SIZE / 2 * 0.9)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_BUSH)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_BUSH_OUTLINE)

        # Draw Acorn
        ax = int(self.acorn_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        ay = int(self.acorn_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        pygame.draw.ellipse(self.screen, self.COLOR_ACORN, (ax - 10, ay - 6, 20, 24))
        pygame.draw.rect(self.screen, self.COLOR_ACORN_STEM, (ax - 3, ay - 12, 6, 8))

    def _render_guards(self):
        for guard in self.guards:
            # Draw vision cone
            guard_px = guard["pos"] * self.TILE_SIZE + self.TILE_SIZE / 2
            
            p1 = guard_px
            
            angle1 = math.atan2(guard["dir"][1], guard["dir"][0]) - guard["vision_angle"]
            p2 = guard_px + guard["vision_range"] * self.TILE_SIZE * np.array([math.cos(angle1), math.sin(angle1)])
            
            angle2 = math.atan2(guard["dir"][1], guard["dir"][0]) + guard["vision_angle"]
            p3 = guard_px + guard["vision_range"] * self.TILE_SIZE * np.array([math.cos(angle2), math.sin(angle2)])

            vision_poly = [p1.tolist(), p2.tolist(), p3.tolist()]
            pygame.gfxdraw.aapolygon(self.screen, vision_poly, self.COLOR_VISION_CONE)
            pygame.gfxdraw.filled_polygon(self.screen, vision_poly, self.COLOR_VISION_CONE)

            # Draw guard body
            gx_px, gy_px = int(guard_px[0]), int(guard_px[1])
            radius = self.TILE_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, gx_px, gy_px, radius, self.COLOR_GUARD)
            pygame.gfxdraw.aacircle(self.screen, gx_px, gy_px, radius, self.COLOR_GUARD_OUTLINE)

            # Draw direction indicator
            dir_end_x = gx_px + int(guard["dir"][0] * radius)
            dir_end_y = gy_px + int(guard["dir"][1] * radius)
            pygame.draw.line(self.screen, self.COLOR_GUARD_OUTLINE, (gx_px, gy_px), (dir_end_x, dir_end_y), 3)

    def _render_player(self):
        px = int(self.squirrel_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        py = int(self.squirrel_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        radius = int(self.TILE_SIZE / 2 * 0.7)
        
        color = self.COLOR_SQUIRREL
        if self.is_hidden:
            color = (
                self.COLOR_SQUIRREL[0],
                self.COLOR_SQUIRREL[1],
                self.COLOR_SQUIRREL[2],
                100,  # Alpha
            )
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, self.COLOR_SQUIRREL_OUTLINE)
            self.screen.blit(temp_surf, (px - radius, py - radius))
        else:
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_SQUIRREL)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_SQUIRREL_OUTLINE)

    def _render_effects(self):
        for effect in self.detection_effects:
            pos_px = effect["pos"] * self.TILE_SIZE
            text = self.font_detection.render("!", True, self.COLOR_DETECTION)
            text_rect = text.get_rect(center=(int(pos_px[0] + self.TILE_SIZE/2), int(pos_px[1] - self.TILE_SIZE/4)))
            self.screen.blit(text, text_rect)

    def _render_ui(self):
        ui_y_pos = self.GRID_HEIGHT * self.TILE_SIZE + (self.UI_HEIGHT / 2) - 9

        # Turns Left
        turns_text = f"Steps Left: {self.MAX_STEPS - self.steps}"
        self._draw_text(turns_text, (10, ui_y_pos), self.font_ui)

        # Detections
        det_text = f"Detections: {self.detections} / {self.MAX_DETECTIONS}"
        det_surf = self.font_ui.render(det_text, True, self.COLOR_TEXT)
        det_rect = det_surf.get_rect(right=self.SCREEN_WIDTH - 10, y=ui_y_pos)
        self._draw_text(det_text, (det_rect.x, det_rect.y), self.font_ui)
        
        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(centerx=self.SCREEN_WIDTH/2, y=ui_y_pos)
        self._draw_text(score_text, (score_rect.x, score_rect.y), self.font_ui)

    def _draw_text(self, text, pos, font):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
        self.screen.blit(text_surf, pos)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Stealthy Squirrel")
    screen_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while not done:
        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = np.array([movement, space_held, shift_held])
        
        # --- Step the environment only on key press for this turn-based game ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                    obs, reward, terminated, truncated, info = env.step(current_action)
                    print(f"Step: {info['steps']}, Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {terminated}")
                    if terminated or truncated:
                        done = True
                    action_taken = True
                    break
        
        if done: break # Exit outer loop if terminated

        # --- Rendering ---
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen_display.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    print("Game Over!")
    print(f"Final Score: {env.score:.2f}")
    
    # Keep window open for a bit to see the final state
    pygame.time.wait(2000)
    env.close()