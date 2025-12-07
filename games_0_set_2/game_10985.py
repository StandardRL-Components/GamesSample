import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:18:09.662944
# Source Brief: brief_00985.md
# Brief Index: 985
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent launches paint projectiles to recreate
    celestial constellations. The agent must master trajectory, timing, and color
    selection to complete the patterns before running out of paint or time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch colorful paint projectiles from your cannon to recreate celestial constellations. "
        "Match the colors and complete the patterns before time or paint runs out."
    )
    user_guide = (
        "Controls: Use ←→ or ↑↓ arrow keys to aim the cannon. "
        "Press space to fire a paint projectile. Press shift to cycle through paint colors."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    INITIAL_TIMER = 2500
    
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (5, 5, 10)
    COLOR_AIM_LINE = (255, 255, 255, 100)
    COLOR_TARGET_INCOMPLETE = (100, 100, 120)
    
    PAINT_COLORS = {
        "Red": (255, 80, 80),
        "Green": (80, 255, 80),
        "Blue": (80, 80, 255),
        "Yellow": (255, 255, 80),
        "Magenta": (255, 80, 255),
        "Cyan": (80, 255, 255),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 20)

        # --- Game State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        
        self.cannon_pos = (0, 0)
        self.aim_angle = 0.0
        
        self.projectiles = []
        self.particles = []
        self.constellations = []
        
        self.paint_inventory = {}
        self.available_colors = list(self.PAINT_COLORS.keys())
        self.selected_color_idx = 0
        
        self.fire_cooldown = 0
        self.prev_shift_state = 0
        self.step_reward = 0.0
        
        self.starfield = []
        self._generate_starfield()

        # Call reset to initialize the game state for the first time
        # self.reset() # This is called by the wrapper/runner, not needed in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.INITIAL_TIMER
        self.game_over = False
        
        self.cannon_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20)
        self.aim_angle = -math.pi / 2  # Start aiming straight up
        
        self.projectiles.clear()
        self.particles.clear()
        
        self._generate_constellations()
        
        self.paint_inventory = {color: 20 for color in self.available_colors}
        self.selected_color_idx = 0
        
        self.fire_cooldown = 0
        self.prev_shift_state = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.step_reward = 0.0
        self.steps += 1
        self.timer -= 1

        self._handle_input(action)
        self._update_game_state()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        # Final reward is calculated inside _check_termination
        reward = self.step_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action
        
        # --- Aiming (Movement) ---
        aim_speed = 0.05  # Radians per step
        if movement == 1:  # Up
            self.aim_angle -= aim_speed
        elif movement == 2:  # Down
            self.aim_angle += aim_speed
        # Note: Left/Right in brief, but Up/Down is more intuitive for angle
        # Let's map left/right to angle as well for completeness.
        elif movement == 3: # Left
            self.aim_angle -= aim_speed
        elif movement == 4: # Right
            self.aim_angle += aim_speed
            
        # Clamp angle to prevent shooting downwards
        self.aim_angle = max(-math.pi + 0.1, min(-0.1, self.aim_angle))

        # --- Color Selection (Shift) ---
        if shift_pressed and not self.prev_shift_state:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.available_colors)
            # sfx: UI_switch_color
        self.prev_shift_state = shift_pressed

        # --- Firing (Space) ---
        if space_pressed and self.fire_cooldown <= 0:
            self._launch_projectile()
            # sfx: launch_projectile

    def _launch_projectile(self):
        color_name = self.available_colors[self.selected_color_idx]
        if self.paint_inventory[color_name] > 0:
            self.paint_inventory[color_name] -= 1
            
            projectile_speed = 8.0
            vel = (
                math.cos(self.aim_angle) * projectile_speed,
                math.sin(self.aim_angle) * projectile_speed
            )
            
            self.projectiles.append({
                "pos": list(self.cannon_pos),
                "vel": vel,
                "color": self.PAINT_COLORS[color_name],
                "color_name": color_name,
                "radius": 6,
                "life": 150 # steps
            })
            self.fire_cooldown = 10 # steps
        else:
            # sfx: out_of_ammo_click
            pass

    def _update_game_state(self):
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        # Update projectiles
        for proj in self.projectiles[:]:
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]
            proj["life"] -= 1

            # Trail particles
            if self.steps % 2 == 0:
                self._create_particles(proj["pos"], proj["color"], 1, 2, -0.2, 20)

            # Collision check
            hit = False
            for const in self.constellations:
                for target in const["targets"]:
                    if target["painted_color"] is None:
                        dist_sq = (proj["pos"][0] - target["pos"][0])**2 + (proj["pos"][1] - target["pos"][1])**2
                        if dist_sq < (proj["radius"] + target["radius"])**2:
                            self._handle_hit(proj, target)
                            hit = True
                            break
                if hit:
                    break
            
            if hit or proj["life"] <= 0 or not (0 < proj["pos"][0] < self.SCREEN_WIDTH and 0 < proj["pos"][1] < self.SCREEN_HEIGHT):
                if not hit: # Wasted projectile
                    self.step_reward -= 0.01
                    self._create_particles(proj["pos"], (100,100,100), 15, 4, -0.5, 40) # sfx: fizzle
                self.projectiles.remove(proj)

        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] - 0.1)
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _handle_hit(self, projectile, target):
        # sfx: hit_target
        if projectile["color_name"] == target["color_req"]:
            target["painted_color"] = projectile["color"]
            self.step_reward += 0.1
            self.score += 10
            self._create_particles(target["pos"], projectile["color"], 20, 5, 0.5, 50, True)
        else: # Wrong color
            self.step_reward -= 0.05
            self.score -= 5
            self._create_particles(target["pos"], (150, 150, 150), 10, 3, 0.2, 30) # sfx: wrong_color_hit
        
        # Check for constellation completion
        constellation = next(c for c in self.constellations if target in c["targets"])
        if not constellation["completed"] and all(t["painted_color"] is not None for t in constellation["targets"]):
            constellation["completed"] = True
            self.score += 100
            self.step_reward += 5.0
            # Power-up: grant more resources
            self.timer += 250
            for color in self.paint_inventory:
                self.paint_inventory[color] += 5
            self.step_reward += 10.0 # Power-up reward
            # sfx: constellation_complete
            
    def _check_termination(self):
        if self.game_over:
            return True

        # Win condition
        if all(c["completed"] for c in self.constellations):
            self.game_over = True
            self.step_reward += 100.0
            self.score += 1000
            return True

        # Lose conditions
        total_paint = sum(self.paint_inventory.values())
        if total_paint <= 0 and not self.projectiles:
            self.game_over = True
            self.step_reward -= 50.0
            return True
        
        if self.timer <= 0:
            self.game_over = True
            self.step_reward -= 50.0
            return True

        return False

    def _get_observation(self):
        # --- Main Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        for star in self.starfield:
            pos, radius, base_alpha = star
            alpha = base_alpha + math.sin(self.steps * 0.05 + pos[0]) * (base_alpha / 2)
            color = (255, 255, 255, max(0, min(255, alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), color)

    def _render_game(self):
        # Render particles (drawn first, behind other elements)
        for p in self.particles:
            alpha = 255 * (p["life"] / p["max_life"])
            color = (*p["color"], max(0, min(255, int(alpha))))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color)
            if p.get("glow", False):
                 pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"] * 1.5), color)

        # Render constellations
        for const in self.constellations:
            for target in const["targets"]:
                pos = (int(target["pos"][0]), int(target["pos"][1]))
                radius = int(target["radius"])
                if target["painted_color"]:
                    # Glow effect for completed targets
                    glow_color = (*target["painted_color"], 50)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 5, glow_color)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 5, glow_color)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, target["painted_color"])
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, target["painted_color"])
                else:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET_INCOMPLETE)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius-1, self.COLOR_TARGET_INCOMPLETE)

        # Render projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], proj["radius"], proj["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], proj["radius"], proj["color"])

        # Render aiming line
        line_length = 50
        end_pos = (
            self.cannon_pos[0] + line_length * math.cos(self.aim_angle),
            self.cannon_pos[1] + line_length * math.sin(self.aim_angle)
        )
        pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, self.cannon_pos, end_pos, 2)
        pygame.gfxdraw.filled_circle(self.screen, int(self.cannon_pos[0]), int(self.cannon_pos[1]), 8, (200, 200, 220))
        pygame.gfxdraw.aacircle(self.screen, int(self.cannon_pos[0]), int(self.cannon_pos[1]), 8, (200, 200, 220))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(1, 1)):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        draw_text(f"Score: {self.score}", self.font_ui, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Timer Bar
        timer_ratio = max(0, self.timer / self.INITIAL_TIMER)
        bar_width = 200
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (bar_x, bar_y, bar_width, bar_height))
        timer_color = (255, 255, 80) if timer_ratio > 0.5 else ((255, 80, 80) if timer_ratio < 0.2 else (255, 165, 0))
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, bar_width * timer_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Paint Inventory
        paint_x = 10
        paint_y = self.SCREEN_HEIGHT - 40
        for i, color_name in enumerate(self.available_colors):
            color_val = self.PAINT_COLORS[color_name]
            count = self.paint_inventory[color_name]
            
            box_size = 30
            box_rect = pygame.Rect(paint_x, paint_y, box_size, box_size)
            
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, (255, 255, 255), box_rect.inflate(6, 6), 2, border_radius=5)

            pygame.draw.rect(self.screen, color_val, box_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, box_rect, 1, border_radius=5)
            
            draw_text(str(count), self.font_small, self.COLOR_TEXT, (paint_x + box_size + 5, paint_y + 8), self.COLOR_TEXT_SHADOW)
            paint_x += box_size + 40

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = all(c["completed"] for c in self.constellations)
            msg = "LEVEL COMPLETE" if win_condition else "GAME OVER"
            draw_text(msg, self.font_title, (255, 255, 80) if win_condition else (255, 80, 80), 
                      (self.SCREEN_WIDTH // 2 - self.font_title.size(msg)[0] // 2, self.SCREEN_HEIGHT // 2 - 50),
                      self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "paint_left": sum(self.paint_inventory.values()),
            "constellations_completed": sum(1 for c in self.constellations if c["completed"]),
        }

    def _generate_starfield(self):
        self.starfield.clear()
        for _ in range(150):
            self.starfield.append(
                (
                    (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                    random.uniform(0.5, 1.5),
                    random.randint(50, 150)
                )
            )

    def _generate_constellations(self):
        self.constellations.clear()
        
        # Level 1: Big Dipper (easy)
        dipper = {
            "name": "Big Dipper",
            "completed": False,
            "targets": [
                {"pos": (100, 150), "radius": 15, "color_req": "Blue", "painted_color": None},
                {"pos": (180, 120), "radius": 15, "color_req": "Blue", "painted_color": None},
                {"pos": (250, 150), "radius": 15, "color_req": "Blue", "painted_color": None},
                {"pos": (320, 130), "radius": 15, "color_req": "Blue", "painted_color": None},
                {"pos": (400, 200), "radius": 15, "color_req": "Red", "painted_color": None},
                {"pos": (450, 250), "radius": 15, "color_req": "Red", "painted_color": None},
                {"pos": (380, 280), "radius": 15, "color_req": "Red", "painted_color": None},
            ]
        }
        self.constellations.append(dipper)

        # Level 2: Orion's Belt (medium)
        orion = {
            "name": "Orion's Belt",
            "completed": False,
            "targets": [
                 {"pos": (500, 80), "radius": 12, "color_req": "Yellow", "painted_color": None},
                 {"pos": (540, 100), "radius": 12, "color_req": "Green", "painted_color": None},
                 {"pos": (580, 120), "radius": 12, "color_req": "Cyan", "painted_color": None},
            ]
        }
        self.constellations.append(orion)

    def _create_particles(self, pos, color, count, radius, speed_mult, life, glow=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color,
                "radius": random.uniform(1, radius),
                "life": random.randint(life // 2, life),
                "max_life": life,
                "glow": glow
            })

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not be used by the evaluation server.
    env = GameEnv()
    obs, info = env.reset()
    
    # Unset the dummy driver to allow for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Constellation Painter")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action
        mov = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: mov = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: mov = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: mov = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: mov = 4
            
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # obs, info = env.reset() # Uncomment to auto-reset

        # Render the observation to the game window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()