
# Generated: 2025-08-27T19:48:57.882781
# Source Brief: brief_02266.md
# Brief Index: 2266

        
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
    user_guide = (
        "Controls: Arrow keys to move cursor. SHIFT to cycle crop type. SPACE to plant or harvest."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate crops against the clock to earn 500 coins in this fast-paced farming simulator."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (50, 80, 60)
    COLOR_UI_BG = (40, 40, 40, 180)
    COLOR_FIELD_BARREN = (110, 80, 50)
    COLOR_FIELD_PLANTED = (30, 60, 30)
    COLOR_FIELD_READY = (204, 173, 23)
    COLOR_CURSOR = (200, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_GOLD = (255, 215, 0)
    COLOR_TIME = (200, 200, 255)

    # Game parameters
    GRID_SIZE = (9, 6)
    WIN_SCORE = 500
    MAX_STEPS = 60
    
    # Crop definitions: [Name, Grow Time (steps), Value, Planted Color, Ready Color]
    CROP_DATA = [
        {"name": "Wheat",   "time": 5,  "value": 10, "color": (245, 222, 179)},
        {"name": "Carrots", "time": 7,  "value": 15, "color": (255, 140, 0)},
        {"name": "Tomatoes","time": 9,  "value": 20, "color": (255, 69, 0)},
        {"name": "Potatoes","time": 11, "value": 25, "color": (160, 82, 45)},
        {"name": "Melons",  "time": 13, "value": 30, "color": (60, 179, 113)},
    ]

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
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        
        # Field and grid layout calculation
        self.field_size = 50
        self.field_gap = 5
        grid_w = self.GRID_SIZE[0] * (self.field_size + self.field_gap) - self.field_gap
        grid_h = self.GRID_SIZE[1] * (self.field_size + self.field_gap) - self.field_gap
        self.grid_offset = (
            (self.screen_size[0] - grid_w) // 2,
            (self.screen_size[1] - grid_h) // 2 - 20,
        )
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.MAX_STEPS

        self.fields = [
            [{"state": "BARREN", "crop_idx": -1, "growth": 0.0} for _ in range(self.GRID_SIZE[1])]
            for _ in range(self.GRID_SIZE[0])
        ]
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.selected_crop_idx = 0
        
        self.last_action = np.zeros(self.action_space.shape)
        self.last_score_reward_milestone = 0
        self.particles = []
        self.action_feedback = [] # To show harvest/plant text popups

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Update Game State ---
        self.steps += 1
        self.timer -= 1
        self._update_crops()
        
        # --- Handle Player Actions ---
        reward += self._handle_actions(action)
        self.last_action = action
        
        # --- Calculate Rewards ---
        new_milestone = self.score // 10
        reward += (new_milestone - self.last_score_reward_milestone) * 1.0
        self.last_score_reward_milestone = new_milestone

        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            self.win = True
        elif self.timer <= 0:
            # Only apply penalty if the player hasn't already won
            if not self.win:
                reward -= 100
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_crops(self):
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                field = self.fields[x][y]
                if field["state"] == "PLANTED":
                    crop = self.CROP_DATA[field["crop_idx"]]
                    # Growth is per step, not time, so increment by 1/time_to_grow
                    field["growth"] += 1.0 / crop["time"]
                    if field["growth"] >= 1.0:
                        field["state"] = "READY"
                        field["growth"] = 1.0

    def _handle_actions(self, action):
        movement, space_raw, shift_raw = action
        space_press = space_raw == 1 and self.last_action[1] == 0
        shift_press = shift_raw == 1 and self.last_action[2] == 0
        
        reward = 0

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right

        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_SIZE[0]
        self.cursor_pos[1] %= self.GRID_SIZE[1]

        # Cycle selected crop
        if shift_press:
            self.selected_crop_idx = (self.selected_crop_idx + 1) % len(self.CROP_DATA)
            # sfx: UI_CYCLE

        # Plant / Harvest
        if space_press:
            cx, cy = self.cursor_pos
            field = self.fields[cx][cy]
            
            if field["state"] == "BARREN":
                # Plant
                field["state"] = "PLANTED"
                field["crop_idx"] = self.selected_crop_idx
                field["growth"] = 0.0
                reward += 0.1
                self._spawn_particles(self.cursor_pos, 10, self.CROP_DATA[self.selected_crop_idx]["color"])
                # sfx: PLANT
            
            elif field["state"] == "READY":
                # Harvest
                crop = self.CROP_DATA[field["crop_idx"]]
                self.score += crop["value"]
                reward += 0.2
                
                # Reset field
                field["state"] = "BARREN"
                field["crop_idx"] = -1
                field["growth"] = 0.0

                self._spawn_particles(self.cursor_pos, 15, self.COLOR_GOLD, "up")
                self._add_action_feedback(f"+{crop['value']}", self.cursor_pos, self.COLOR_GOLD)
                # sfx: HARVEST_COIN

        return reward

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_fields()
        self._render_cursor()
        self._update_and_render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_fields(self):
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                field = self.fields[x][y]
                px = self.grid_offset[0] + x * (self.field_size + self.field_gap)
                py = self.grid_offset[1] + y * (self.field_size + self.field_gap)
                rect = pygame.Rect(px, py, self.field_size, self.field_size)

                color = self.COLOR_FIELD_BARREN
                if field["state"] == "PLANTED":
                    color = self.COLOR_FIELD_PLANTED
                elif field["state"] == "READY":
                    color = self.COLOR_FIELD_READY
                
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                if field["state"] == "PLANTED" or field["state"] == "READY":
                    crop = self.CROP_DATA[field["crop_idx"]]
                    crop_size = int(max(5, (self.field_size * 0.8) * field["growth"]))
                    crop_rect = pygame.Rect(0, 0, crop_size, crop_size)
                    crop_rect.center = rect.center
                    pygame.draw.rect(self.screen, crop["color"], crop_rect, border_radius=crop_size//3)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px = self.grid_offset[0] + cx * (self.field_size + self.field_gap)
        py = self.grid_offset[1] + cy * (self.field_size + self.field_gap)
        rect = pygame.Rect(px - 4, py - 4, self.field_size + 8, self.field_size + 8)

        # Pulsing alpha effect for cursor
        alpha = 128 + 127 * math.sin(self.steps * 0.3)
        cursor_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, (*self.COLOR_CURSOR, alpha), cursor_surf.get_rect(), 4, border_radius=8)
        self.screen.blit(cursor_surf, rect.topleft)

    def _render_ui(self):
        # Top UI (Score and Timer)
        score_text = self.font_m.render(f"{self.score}", True, self.COLOR_GOLD)
        pygame.gfxdraw.filled_circle(self.screen, 30, 30, 16, self.COLOR_UI_BG)
        pygame.gfxdraw.aacircle(self.screen, 30, 30, 16, self.COLOR_GOLD)
        self.screen.blit(score_text, (55, 20))

        time_text = self.font_m.render(f"{self.timer}", True, self.COLOR_TIME)
        pygame.gfxdraw.filled_circle(self.screen, self.screen_size[0] - 30, 30, 16, self.COLOR_UI_BG)
        pygame.gfxdraw.aacircle(self.screen, self.screen_size[0] - 30, 30, 16, self.COLOR_TIME)
        time_text_rect = time_text.get_rect(right=self.screen_size[0] - 55, centery=30)
        self.screen.blit(time_text, time_text_rect)

        # Bottom UI (Selected Crop)
        ui_panel_rect = pygame.Rect(0, self.screen_size[1] - 60, self.screen_size[0], 60)
        s = pygame.Surface(ui_panel_rect.size, pygame.SRCALPHA)
        s.fill((*self.COLOR_UI_BG[:3], 200))
        self.screen.blit(s, ui_panel_rect.topleft)

        crop = self.CROP_DATA[self.selected_crop_idx]
        crop_icon_rect = pygame.Rect(20, ui_panel_rect.y + 15, 30, 30)
        pygame.draw.rect(self.screen, crop["color"], crop_icon_rect, border_radius=5)
        
        name_text = self.font_m.render(crop["name"], True, self.COLOR_TEXT)
        self.screen.blit(name_text, (65, ui_panel_rect.y + 20))

        info_text_str = f"Grows in: {crop['time']}s  |  Sells for: {crop['value']}"
        info_text = self.font_s.render(info_text_str, True, self.COLOR_TEXT)
        info_text_rect = info_text.get_rect(right=self.screen_size[0] - 20, centery=ui_panel_rect.centery)
        self.screen.blit(info_text, info_text_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "TIME'S UP!"
            color = self.COLOR_GOLD if self.win else self.COLOR_TIME
            
            end_text = pygame.font.Font(None, 80).render(message, True, color)
            end_rect = end_text.get_rect(center=(self.screen_size[0] / 2, self.screen_size[1] / 2))
            self.screen.blit(end_text, end_rect)

    def _spawn_particles(self, grid_pos, count, color, direction=None):
        px = self.grid_offset[0] + grid_pos[0] * (self.field_size + self.field_gap) + self.field_size / 2
        py = self.grid_offset[1] + grid_pos[1] * (self.field_size + self.field_gap) + self.field_size / 2
        
        for _ in range(count):
            if direction == "up":
                angle = random.uniform(-math.pi / 2 - 0.5, -math.pi / 2 + 0.5)
                speed = random.uniform(2, 4)
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
            
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 40)
            self.particles.append({"pos": [px, py], "vel": vel, "life": lifespan, "color": color})

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.05 # Gravity
            p["life"] -= 1
            
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p["life"] / 8))
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                pygame.draw.circle(self.screen, p["color"], pos, size)

    def _add_action_feedback(self, text, grid_pos, color):
        px = self.grid_offset[0] + grid_pos[0] * (self.field_size + self.field_gap) + self.field_size / 2
        py = self.grid_offset[1] + grid_pos[1] * (self.field_size + self.field_gap)
        self.action_feedback.append({"text": text, "pos": [px, py], "life": 30, "color": color})

    def _render_action_feedback(self):
        for fb in self.action_feedback[:]:
            fb["pos"][1] -= 0.5
            fb["life"] -= 1
            if fb["life"] <= 0:
                self.action_feedback.remove(fb)
            else:
                alpha = max(0, min(255, int(255 * (fb["life"] / 30))))
                text_surf = self.font_s.render(fb["text"], True, fb["color"])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=fb["pos"])
                self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cursor_pos": self.cursor_pos,
            "selected_crop": self.CROP_DATA[self.selected_crop_idx]["name"]
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = env.action_space.sample() # Start with a random action
        action[0] = 0 # No movement by default
        action[1] = 0 # No space
        action[2] = 0 # No shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Display the rendered frame
        # The observation is (H, W, C), but pygame wants (W, H)
        # and surfarray.make_surface expects (W, H, C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise Exception
            display_surf.blit(surf, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode(env.screen_size)
            display_surf.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()