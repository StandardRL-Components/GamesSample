import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:00:37.918066
# Source Brief: brief_00154.md
# Brief Index: 154
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gearshift Velocity: A real-time puzzle game Gymnasium environment.

    The goal is to link chains of 3 same-colored gears to increase the RPM
    of a central platform to 100 within a 45-second time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Link chains of three same-colored gears to increase the RPM of a central platform to 100 within the time limit."
    user_guide = "Use ←→ arrow keys to select a gear. Press space to link gears into a chain and shift to use a boost on the selected gear."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants and Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.GAME_DURATION_SECONDS = 45
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.WIN_RPM = 100.0
        self.NUM_GEARS = 5
        self.MAX_CHAIN_LENGTH = 3
        self.INITIAL_POWERUPS = 3

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (30, 40, 50)
        self.GEAR_COLORS = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 120, 255),
            "yellow": (255, 255, 80),
            "purple": (200, 80, 255),
        }
        self.COLOR_NAMES = list(self.GEAR_COLORS.keys())
        self.COLOR_PLATFORM = (220, 220, 240)
        self.COLOR_GLOW = (150, 200, 255)
        self.COLOR_SELECT = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SHADOW = (10, 10, 10)
        self.COLOR_SUPERCHARGED = (255, 165, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Initialization ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0
        self.platform_rpm = 0.0
        self.powerups_left = 0
        self.gears = []
        self.linked_gears_indices = []
        self.selected_gear_idx = 0
        self.particles = []

        # Action handling state
        self.prev_movement = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        # This check is for development and ensures the implementation matches the spec.
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.platform_rpm = 0.0
        self.powerups_left = self.INITIAL_POWERUPS
        
        self.linked_gears_indices = []
        self.selected_gear_idx = 0
        self.particles = []

        self.prev_movement = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._initialize_gears()

        return self._get_observation(), self._get_info()

    def _initialize_gears(self):
        self.gears = []
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        layout_radius = 120
        
        for i in range(self.NUM_GEARS):
            angle = (i / self.NUM_GEARS) * 2 * math.pi + (math.pi / 2)
            x = center_x + layout_radius * math.cos(angle)
            y = center_y + layout_radius * math.sin(angle)
            
            color_name = self.np_random.choice(self.COLOR_NAMES)
            
            self.gears.append({
                "pos": (x, y),
                "radius": 35,
                "teeth": 10,
                "color_name": color_name,
                "color_value": self.GEAR_COLORS[color_name],
                "rpm": self.np_random.uniform(10, 50),
                "angle": self.np_random.uniform(0, 360),
                "is_supercharged": False,
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        reward = self._handle_input(action)

        # --- Update Game State ---
        previous_rpm = self.platform_rpm
        self._update_game_state()
        
        # Continuous reward for RPM increase
        rpm_increase = self.platform_rpm - previous_rpm
        if rpm_increase > 0:
            reward += 0.1 * rpm_increase

        self.steps += 1
        self.timer -= 1
        
        # --- Check Termination ---
        terminated = self.platform_rpm >= self.WIN_RPM or self.timer <= 0
        if terminated:
            self.game_over = True
            if self.platform_rpm >= self.WIN_RPM:
                reward += 100  # Victory bonus
            else:
                reward -= 100  # Timeout penalty

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        event_reward = 0

        # Detect rising edge for presses to avoid repeated actions
        left_press = movement == 3 and self.prev_movement != 3
        right_press = movement == 4 and self.prev_movement != 4
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        if left_press:
            self.selected_gear_idx = (self.selected_gear_idx - 1 + self.NUM_GEARS) % self.NUM_GEARS
        if right_press:
            self.selected_gear_idx = (self.selected_gear_idx + 1) % self.NUM_GEARS

        if space_press:
            event_reward += self._link_gear()

        if shift_press:
            self._use_powerup()

        self.prev_movement = movement
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return event_reward

    def _link_gear(self):
        if self.selected_gear_idx not in self.linked_gears_indices:
            self.linked_gears_indices.append(self.selected_gear_idx)
            # sound effect: 'link_gear.wav'
            if len(self.linked_gears_indices) == self.MAX_CHAIN_LENGTH:
                return self._check_and_process_chain()
        return 0

    def _check_and_process_chain(self):
        gears_in_chain = [self.gears[i] for i in self.linked_gears_indices]
        first_color = gears_in_chain[0]["color_name"]
        
        if all(g["color_name"] == first_color for g in gears_in_chain):
            # Successful chain
            total_rpm_boost = sum(g["rpm"] for g in gears_in_chain)
            self.platform_rpm = min(self.WIN_RPM, self.platform_rpm + total_rpm_boost)
            # sound effect: 'chain_success.wav'
            self.linked_gears_indices.clear()
            return 5.0  # Event-based reward for a successful chain
        else:
            # Failed chain
            # sound effect: 'chain_fail.wav'
            self.linked_gears_indices.clear()
            return -1.0 # Small penalty for a failed chain

    def _use_powerup(self):
        if self.powerups_left > 0:
            gear = self.gears[self.selected_gear_idx]
            if not gear["is_supercharged"]:
                self.powerups_left -= 1
                gear["rpm"] *= 2
                gear["is_supercharged"] = True
                # sound effect: 'powerup.wav'
                # Particle effect for power-up
                for _ in range(30):
                    self.particles.append(Particle(gear["pos"], self.COLOR_SUPERCHARGED, self.np_random))

    def _update_game_state(self):
        # Update gear rotations
        for gear in self.gears:
            # RPM is revs per minute. Angle change per step (at 60 FPS) is:
            # (RPM / 60 sec/min) * 360 deg/rev / 60 frames/sec = RPM * 0.1 deg/frame
            gear["angle"] = (gear["angle"] + gear["rpm"] * 0.1) % 360

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game(self):
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2

        # Render connecting lines
        if len(self.linked_gears_indices) > 0:
            points = [self.gears[i]["pos"] for i in self.linked_gears_indices]
            if len(points) > 1:
                color = self.gears[self.linked_gears_indices[0]]["color_value"]
                pygame.draw.aalines(self.screen, color, False, points, 2)

        # Render platform
        glow_alpha = min(255, int(50 + (self.platform_rpm / self.WIN_RPM) * 205))
        glow_radius = int(40 + (self.platform_rpm / self.WIN_RPM) * 40)
        
        for i in range(5):
            alpha = glow_alpha * (1 - i / 5) ** 2
            radius = glow_radius * (1 - i / 10)
            if alpha > 0 and radius > 0:
                pygame.gfxdraw.aacircle(self.screen, cx, cy, int(radius), (*self.COLOR_GLOW, int(alpha)))
        
        pygame.gfxdraw.aacircle(self.screen, cx, cy, 40, self.COLOR_PLATFORM)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 40, self.COLOR_PLATFORM)
        
        # Render gears
        for i, gear in enumerate(self.gears):
            is_selected = (i == self.selected_gear_idx)
            is_linked = (i in self.linked_gears_indices)
            self._draw_gear(gear, is_selected, is_linked)
            
        # Render particles
        for p in self.particles:
            p.draw(self.screen)

    def _draw_gear(self, gear, is_selected, is_linked):
        x, y = int(gear["pos"][0]), int(gear["pos"][1])
        radius = gear["radius"]
        angle_rad = math.radians(gear["angle"])
        
        # Supercharge effect
        if gear["is_supercharged"]:
            for i in range(3):
                alpha = 100 - i * 30
                r = radius + 5 + i * 3
                pygame.gfxdraw.aacircle(self.screen, x, y, r, (*self.COLOR_SUPERCHARGED, alpha))

        # Selection highlight
        if is_selected:
            pygame.gfxdraw.aacircle(self.screen, x, y, radius + 6, self.COLOR_SELECT)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius + 5, self.COLOR_SELECT)
        
        # Linked highlight
        if is_linked:
            pygame.gfxdraw.aacircle(self.screen, x, y, radius + 3, gear["color_value"])

        # Gear teeth
        points = []
        outer_r, inner_r = radius, radius * 0.8
        for i in range(gear["teeth"] * 2):
            r = outer_r if i % 2 == 0 else inner_r
            current_angle = (i / (gear["teeth"] * 2)) * 2 * math.pi + angle_rad
            px = x + r * math.cos(current_angle)
            py = y + r * math.sin(current_angle)
            points.append((px, py))
        
        pygame.gfxdraw.aapolygon(self.screen, points, gear["color_value"])
        pygame.gfxdraw.filled_polygon(self.screen, points, gear["color_value"])
        
        # Gear center
        pygame.gfxdraw.aacircle(self.screen, x, y, int(radius * 0.4), self.COLOR_BG)
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius * 0.4), self.COLOR_BG)
        
    def _render_ui(self):
        # Timer
        time_text = f"{self.timer / self.FPS:.1f}"
        self._draw_text(time_text, self.font_medium, (self.WIDTH - 10, 10), "topright")
        
        # Platform RPM
        rpm_text = f"{self.platform_rpm:.1f}"
        self._draw_text(rpm_text, self.font_medium, (self.WIDTH // 2, self.HEIGHT // 2), "center")
        rpm_label = "RPM"
        self._draw_text(rpm_label, self.font_small, (self.WIDTH // 2, self.HEIGHT // 2 + 20), "center")
        
        # Power-ups
        self._draw_text("BOOST", self.font_small, (10, 10), "topleft")
        for i in range(self.INITIAL_POWERUPS):
            color = self.COLOR_SUPERCHARGED if i < self.powerups_left else self.COLOR_GRID
            pos = (20 + i * 25, 40)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
            
        # Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.platform_rpm >= self.WIN_RPM else "TIME OUT"
            self._draw_text(message, self.font_large, (self.WIDTH // 2, self.HEIGHT // 2), "center")

    def _draw_text(self, text, font, pos, anchor="topleft"):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_SHADOW)
        text_rect = text_surf.get_rect()
        
        if anchor == "topleft":
            text_rect.topleft = pos
        elif anchor == "topright":
            text_rect.topright = pos
        elif anchor == "center":
            text_rect.center = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "platform_rpm": self.platform_rpm,
            "timer": self.timer / self.FPS,
            "powerups_left": self.powerups_left
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

class Particle:
    def __init__(self, pos, color, np_random):
        self.x, self.y = pos
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 60  # 1 second life at 60 FPS
        self.color = color
        self.radius = self.np_random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / 60))
            color = (*self.color, alpha)
            pos = (int(self.x), int(self.y))
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), color)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), color)

# Example of how to run the environment
if __name__ == '__main__':
    # To run with display, comment out the os.environ line at the top of the file
    try:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # For human play
        pygame.display.set_caption("Gearshift Velocity")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        
        total_reward = 0
        
        while not done:
            # Map keyboard keys to the MultiDiscrete action space for human play
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render the observation to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    
        print(f"Game Over! Final Score: {info['score']:.2f}, Final RPM: {info['platform_rpm']:.1f}")
        
        # Keep the final screen visible for a moment
        pygame.time.wait(2000)
        
        env.close()
    except pygame.error as e:
        print("\nPygame error detected. To run this script with a display,")
        print("you might need to comment out the following line at the top of the file:")
        print('os.environ.setdefault("SDL_VIDEODRIVER", "dummy")\n')
        print(f"Original error: {e}")