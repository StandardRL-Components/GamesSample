
# Generated: 2025-08-28T05:10:03.989928
# Source Brief: brief_05490.md
# Brief Index: 5490

        
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
    An arcade action game where the player slices falling fruit with a virtual blade.
    The goal is to slice a target number of fruits before missing too many.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to slice."
    )

    game_description = (
        "Slice falling fruit with a blade to score points. Reach 50 points to win, but don't miss 10 fruit!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 50
    LOSE_MISSES = 10
    MAX_STEPS = 3000 # Increased for a longer game experience

    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_TRAIL = (255, 255, 255)
    COLOR_CURSOR = (200, 200, 255, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SHADOW = (20, 20, 20)
    FRUIT_COLORS = {
        "apple": (220, 40, 40),
        "banana": (255, 225, 50),
        "kiwi": (100, 180, 70),
        "orange": (255, 165, 0),
        "plum": (140, 80, 180),
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
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)
        
        self.game_state = {}
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_state = {
            "steps": 0,
            "score": 0,
            "missed_fruits": 0,
            "cursor_pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float),
            "fruits": [],
            "particles": [],
            "slice_trails": [],
            "last_space_held": False,
            "base_fruit_speed": 2.0,
            "current_fruit_speed": 2.0,
            "fruit_spawn_timer": 0,
            "fruit_spawn_rate": 30, # Spawn every N frames on average
            "game_over_message": ""
        }
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        # --- Update Game Logic ---
        self.game_state["steps"] += 1
        
        self._handle_input(movement, space_held)
        
        sliced_this_step = self._update_slice(space_held)
        if sliced_this_step > 0:
            reward += sliced_this_step # +1 per fruit
            if sliced_this_step > 1:
                reward += 2 # Combo bonus
                # TODO: Play "combo" sound

        self._update_fruits()
        self._update_particles()
        self._update_slice_trails()
        self._spawn_fruits()
        
        # Continuous negative reward for existing fruits
        reward -= 0.01 * len(self.game_state["fruits"])

        # --- Check Termination ---
        if self.game_state["score"] >= self.WIN_SCORE:
            terminated = True
            reward += 100
            self.game_state["game_over_message"] = "YOU WIN!"
        elif self.game_state["missed_fruits"] >= self.LOSE_MISSES:
            terminated = True
            reward -= 100
            self.game_state["game_over_message"] = "GAME OVER"
        elif self.game_state["steps"] >= self.MAX_STEPS:
            terminated = True
            self.game_state["game_over_message"] = "TIME UP"
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        cursor_speed = 15.0
        if movement == 1: # Up
            self.game_state["cursor_pos"][1] -= cursor_speed
        elif movement == 2: # Down
            self.game_state["cursor_pos"][1] += cursor_speed
        elif movement == 3: # Left
            self.game_state["cursor_pos"][0] -= cursor_speed
        elif movement == 4: # Right
            self.game_state["cursor_pos"][0] += cursor_speed

        self.game_state["cursor_pos"][0] = np.clip(self.game_state["cursor_pos"][0], 0, self.SCREEN_WIDTH)
        self.game_state["cursor_pos"][1] = np.clip(self.game_state["cursor_pos"][1], 0, self.SCREEN_HEIGHT)
    
    def _update_slice(self, space_held):
        sliced_count = 0
        if space_held and not self.game_state["last_space_held"]:
            # TODO: Play "swoosh" sound
            slice_angle = self.np_random.uniform(0, 2 * math.pi)
            slice_length = 120
            p1 = self.game_state["cursor_pos"] + np.array([math.cos(slice_angle), math.sin(slice_angle)]) * slice_length / 2
            p2 = self.game_state["cursor_pos"] - np.array([math.cos(slice_angle), math.sin(slice_angle)]) * slice_length / 2
            
            self.game_state["slice_trails"].append({"p1": p1, "p2": p2, "life": 10})

            sliced_fruits_indices = []
            for i, fruit in enumerate(self.game_state["fruits"]):
                if self._line_circle_collision(p1, p2, fruit["pos"], fruit["radius"]):
                    sliced_fruits_indices.append(i)

            # Iterate backwards to avoid index errors on pop
            for i in sorted(sliced_fruits_indices, reverse=True):
                fruit = self.game_state["fruits"].pop(i)
                sliced_count += 1
                self.game_state["score"] += 1
                self._create_particles(fruit["pos"], fruit["color"])
                # TODO: Play "slice" sound

                # Update difficulty every 5 fruits
                if self.game_state["score"] > 0 and self.game_state["score"] % 5 == 0:
                    self.game_state["current_fruit_speed"] += 0.2
        
        self.game_state["last_space_held"] = space_held
        return sliced_count

    def _update_fruits(self):
        fruits_to_remove = []
        for i, fruit in enumerate(self.game_state["fruits"]):
            fruit["pos"] += fruit["vel"]
            fruit["angle"] += fruit["rot_speed"]
            if fruit["pos"][1] > self.SCREEN_HEIGHT + fruit["radius"]:
                fruits_to_remove.append(i)
                self.game_state["missed_fruits"] += 1
                # TODO: Play "miss" sound

        for i in sorted(fruits_to_remove, reverse=True):
            self.game_state["fruits"].pop(i)

    def _spawn_fruits(self):
        self.game_state["fruit_spawn_timer"] -= 1
        if self.game_state["fruit_spawn_timer"] <= 0:
            self.game_state["fruit_spawn_timer"] = self.np_random.integers(
                low=int(self.game_state["fruit_spawn_rate"] * 0.5), 
                high=int(self.game_state["fruit_spawn_rate"] * 1.5)
            )
            
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            radius = self.np_random.integers(15, 25)
            pos = np.array([self.np_random.uniform(radius, self.SCREEN_WIDTH - radius), -radius], dtype=float)
            vel = np.array([self.np_random.uniform(-1, 1), self.game_state["current_fruit_speed"]], dtype=float)
            
            self.game_state["fruits"].append({
                "pos": pos,
                "vel": vel,
                "radius": radius,
                "color": self.FRUIT_COLORS[fruit_type],
                "angle": 0,
                "rot_speed": self.np_random.uniform(-0.1, 0.1)
            })

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.game_state["particles"].append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        self.game_state["particles"] = [p for p in self.game_state["particles"] if p["life"] > 0]
        for p in self.game_state["particles"]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["life"] -= 1

    def _update_slice_trails(self):
        self.game_state["slice_trails"] = [s for s in self.game_state["slice_trails"] if s["life"] > 0]
        for s in self.game_state["slice_trails"]:
            s["life"] -= 1

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a simple gradient
        rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, rect)
        
        num_strips = 20
        strip_height = self.SCREEN_HEIGHT / num_strips
        for i in range(num_strips):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * (i / num_strips)
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * (i / num_strips)
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * (i / num_strips)
            pygame.draw.rect(self.screen, (r, g, b), (0, i * strip_height, self.SCREEN_WIDTH, strip_height))

    def _render_game(self):
        # Render particles
        for p in self.game_state["particles"]:
            alpha = max(0, 255 * (p["life"] / 30))
            s = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["color"], alpha), (3, 3), 3)
            self.screen.blit(s, (int(p["pos"][0] - 3), int(p["pos"][1] - 3)))

        # Render fruits
        for fruit in self.game_state["fruits"]:
            x, y = int(fruit["pos"][0]), int(fruit["pos"][1])
            r = fruit["radius"]
            # Shadow
            pygame.gfxdraw.filled_circle(self.screen, x + 3, y + 3, r, (*self.COLOR_SHADOW, 100))
            # Main fruit body
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, fruit["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, fruit["color"])
            # Shine
            shine_x = int(x + r * 0.3 * math.cos(fruit["angle"] + math.pi / 4))
            shine_y = int(y - r * 0.3 * math.sin(fruit["angle"] + math.pi / 4))
            pygame.gfxdraw.filled_circle(self.screen, shine_x, shine_y, int(r * 0.2), (255, 255, 255, 150))
            
        # Render slice trails
        for s in self.game_state["slice_trails"]:
            alpha = max(0, 255 * (s["life"] / 10))
            width = int(10 * (s["life"] / 10))
            self._draw_aa_thick_line(s["p1"], s["p2"], width, (*self.COLOR_TRAIL, alpha))

        # Render cursor
        cursor_surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(cursor_surf, 16, 16, 10, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(cursor_surf, 16, 16, 10, (255,255,255,200))
        self.screen.blit(cursor_surf, (int(self.game_state["cursor_pos"][0] - 16), int(self.game_state["cursor_pos"][1] - 16)))
        
    def _render_ui(self):
        score_text = f"SCORE: {self.game_state['score']}"
        miss_text = f"MISSED: {self.game_state['missed_fruits']}/{self.LOSE_MISSES}"
        
        self._draw_text(score_text, (15, 10), self.font_small)
        self._draw_text(miss_text, (15, 40), self.font_small)

        if self.game_state["game_over_message"]:
            self._draw_text(self.game_state["game_over_message"], 
                            (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), 
                            self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        shadow_color = (max(0, color[0]-150), max(0, color[1]-150), max(0, color[2]-150))
        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def _draw_aa_thick_line(self, p1, p2, width, color):
        if width <= 1:
            pygame.draw.aaline(self.screen, color, p1, p2)
            return

        center_L1 = (p1 + p2) / 2
        length = np.linalg.norm(p2 - p1)
        angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        
        UL = (center_L1[0] + (length/2.) * math.cos(angle) - (width/2.) * math.sin(angle),
              center_L1[1] + (width/2.) * math.cos(angle) + (length/2.) * math.sin(angle))
        UR = (center_L1[0] - (length/2.) * math.cos(angle) - (width/2.) * math.sin(angle),
              center_L1[1] + (width/2.) * math.cos(angle) - (length/2.) * math.sin(angle))
        BL = (center_L1[0] + (length/2.) * math.cos(angle) + (width/2.) * math.sin(angle),
              center_L1[1] - (width/2.) * math.cos(angle) + (length/2.) * math.sin(angle))
        BR = (center_L1[0] - (length/2.) * math.cos(angle) + (width/2.) * math.sin(angle),
              center_L1[1] - (width/2.) * math.cos(angle) - (length/2.) * math.sin(angle))
        
        points = [UL, UR, BR, BL]
        
        # Use a surface for transparency
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        
        size = (max_x - min_x, max_y - min_y)
        if size[0] <= 0 or size[1] <= 0: return

        surf = pygame.Surface(size, pygame.SRCALPHA)
        local_points = [(p[0] - min_x, p[1] - min_y) for p in points]
        
        pygame.gfxdraw.aapolygon(surf, local_points, color)
        pygame.gfxdraw.filled_polygon(surf, local_points, color)
        self.screen.blit(surf, (min_x, min_y))


    def _line_circle_collision(self, p1, p2, circle_center, circle_radius):
        # Based on https://stackoverflow.com/a/1079478
        p1 = p1 - circle_center
        p2 = p2 - circle_center
        
        d = p2 - p1
        dr2 = np.dot(d, d)
        if dr2 == 0: # Points are the same
            return np.linalg.norm(p1) <= circle_radius
        
        t = np.dot(-p1, d) / dr2
        t = np.clip(t, 0, 1)
        
        closest_point = p1 + t * d
        return np.linalg.norm(closest_point) <= circle_radius

    def _get_info(self):
        return {
            "score": self.game_state["score"],
            "steps": self.game_state["steps"],
            "missed_fruits": self.game_state["missed_fruits"],
            "fruit_count": len(self.game_state["fruits"]),
        }

    def close(self):
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = env.SCREEN_WIDTH, env.SCREEN_HEIGHT
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Fruit Slicer")
    
    terminated = False
    running = True
    clock = pygame.time.Clock()

    while running:
        movement = 0 # none
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, 0] # shift is unused

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated:
            # Simple reset on key press after game over
            if any(keys):
                obs, info = env.reset()
                terminated = False

        clock.tick(30) # Run at 30 FPS

    env.close()