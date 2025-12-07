import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:43:34.305366
# Source Brief: brief_00587.md
# Brief Index: 587
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player survives waves of scrap robots.
    The player launches size-changing, elementally-charged scraps from a fixed platform
    to destroy incoming robots.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive waves of scrap robots by launching elementally-charged projectiles from your platform. "
        "Charge your shots and combine elements to unleash devastating combos."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim your reticle. Hold space to charge your "
        "scrap projectile and release to fire. Press shift to cycle between unlocked elements."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PLATFORM = (40, 50, 60)
    COLOR_PLAYER = (255, 128, 0) # Orange
    COLOR_RETICLE = (255, 255, 255)
    COLOR_ROBOT = (100, 110, 120)
    COLOR_ROBOT_DAMAGED = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    ELEMENT_COLORS = {
        "none": (180, 180, 180),
        "fire": (255, 100, 0),
        "ice": (0, 180, 255),
        "toxic": (100, 255, 50),
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.render_mode = render_mode

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        # Player
        self.player_health = 0
        self.player_platform_rect = None
        self.aim_reticle_pos = None
        self.scrap_charge = 0.0
        self.last_space_state = False
        self.last_shift_state = False

        # Elements
        self.elements = ["fire", "ice", "toxic"]
        self.unlocked_elements = []
        self.current_element_idx = 0
        
        # Combos
        self.unlocked_combos = []

        # Projectiles
        self.scraps = []
        self.particles = []

        # Enemies
        self.robots = []
        self.wave_number = 0
        self.wave_transition_timer = 0
        
        # Initialize state variables
        # self.reset() # No need to call reset in __init__
        # self.validate_implementation() # Not needed for final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_health = 100.0
        self.player_platform_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 60, self.SCREEN_WIDTH, 60)
        self.aim_reticle_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.scrap_charge = 0.0
        self.last_space_state = False
        self.last_shift_state = False

        # Elements & Combos
        self.unlocked_elements = ["fire"]
        self.current_element_idx = 0
        self.unlocked_combos = []

        # Game entities
        self.scraps = []
        self.particles = []
        self.robots = []
        
        # Wave management
        self.wave_number = 0
        self.wave_transition_timer = self.FPS * 3 # 3 seconds before first wave
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        terminated = self.player_health <= 0

        if not terminated:
            self._handle_input(action)
            self._update_game_state()
            self._handle_collisions()
            self._update_wave_logic()

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        terminated = self.player_health <= 0 or (self.wave_number > self.MAX_WAVES)

        # Terminal rewards
        if terminated and self.player_health <= 0:
            self.reward_this_step = -100.0
        elif terminated and self.wave_number > self.MAX_WAVES:
            self.reward_this_step = 100.0
            
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Aiming Reticle Movement ---
        reticle_speed = 8
        if movement == 1: self.aim_reticle_pos.y -= reticle_speed
        elif movement == 2: self.aim_reticle_pos.y += reticle_speed
        elif movement == 3: self.aim_reticle_pos.x -= reticle_speed
        elif movement == 4: self.aim_reticle_pos.x += reticle_speed
        
        self.aim_reticle_pos.x = np.clip(self.aim_reticle_pos.x, 0, self.SCREEN_WIDTH)
        self.aim_reticle_pos.y = np.clip(self.aim_reticle_pos.y, 0, self.player_platform_rect.top)

        # --- Scrap Charging (Space) ---
        if space_pressed:
            self.scrap_charge = min(1.0, self.scrap_charge + 0.05)
        
        # --- Scrap Launching (Space Release) ---
        if not space_pressed and self.last_space_state and self.scrap_charge > 0.1:
            # sfx: scrap_launch.wav
            player_center = self.player_platform_rect.centerx
            start_pos = pygame.Vector2(player_center, self.player_platform_rect.top - 10)
            direction = (self.aim_reticle_pos - start_pos).normalize()
            
            size = 5 + 25 * self.scrap_charge
            damage = 10 + 90 * self.scrap_charge
            speed = 15 - 5 * self.scrap_charge
            element = self.unlocked_elements[self.current_element_idx] if self.unlocked_elements else "none"

            self.scraps.append({
                "pos": start_pos, "vel": direction * speed, "size": size,
                "element": element, "damage": damage, "angle": 0
            })
            self.scrap_charge = 0.0

        # --- Element Cycling (Shift Press) ---
        if shift_pressed and not self.last_shift_state and len(self.unlocked_elements) > 1:
            # sfx: element_cycle.wav
            self.current_element_idx = (self.current_element_idx + 1) % len(self.unlocked_elements)

        self.last_space_state = space_pressed
        self.last_shift_state = shift_pressed

    def _update_game_state(self):
        # Update scraps
        for scrap in self.scraps:
            scrap["pos"] += scrap["vel"]
            scrap["angle"] = (scrap["angle"] + scrap["vel"].length() * 2) % 360
        self.scraps = [s for s in self.scraps if self.screen.get_rect().collidepoint(s["pos"])]

        # Update robots
        for robot in self.robots:
            robot["pos"].y += robot["speed"]
            if robot["pos"].y > self.player_platform_rect.top:
                # sfx: player_damage.wav
                self.player_health -= robot["damage"]
                self.reward_this_step -= 0.5
                self._create_explosion(robot["pos"], 20, self.COLOR_ROBOT_DAMAGED)
                robot["active"] = False # Mark for removal
            
            if "damage_timer" in robot:
                robot["damage_timer"] -= 1
                if robot["damage_timer"] <= 0:
                    del robot["damage_timer"]
            
            if "last_hit_timer" in robot:
                robot["last_hit_timer"] -= 1
                if robot["last_hit_timer"] <= 0:
                    del robot["last_hit_timer"]
                    del robot["last_element_hit"]

        self.robots = [r for r in self.robots if r["active"]]

        # Update particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _handle_collisions(self):
        for scrap in self.scraps:
            scrap_rect = pygame.Rect(scrap["pos"].x - scrap["size"]/2, scrap["pos"].y - scrap["size"]/2, scrap["size"], scrap["size"])
            for robot in self.robots:
                robot_rect = pygame.Rect(robot["pos"].x - robot["size"]/2, robot["pos"].y - robot["size"]/2, robot["size"], robot["size"])
                if scrap_rect.colliderect(robot_rect):
                    # sfx: robot_hit.wav
                    self.reward_this_step += 0.1
                    robot["health"] -= scrap["damage"]
                    robot["damage_timer"] = 5 # frames to show damage color
                    self._create_explosion(scrap["pos"], int(scrap["size"]), self.ELEMENT_COLORS[scrap["element"]])
                    
                    # Combo check
                    if "shatter" in self.unlocked_combos and "last_element_hit" in robot and robot["last_element_hit"] != scrap["element"]:
                        if (robot["last_element_hit"] == "fire" and scrap["element"] == "ice") or \
                           (robot["last_element_hit"] == "ice" and scrap["element"] == "fire"):
                            # sfx: combo_shatter.wav
                            self.reward_this_step += 5.0
                            self._create_explosion(robot["pos"], 50, self.ELEMENT_COLORS["ice"])
                            # Area damage
                            for other_robot in self.robots:
                                if other_robot is not robot:
                                    dist = robot["pos"].distance_to(other_robot["pos"])
                                    if dist < 80:
                                        other_robot["health"] -= 50 # Combo damage
                            robot["health"] -= 100 # Extra damage to target

                    robot["last_element_hit"] = scrap["element"]
                    robot["last_hit_timer"] = self.FPS * 2 # 2 seconds to land another element

                    if robot["health"] <= 0:
                        # sfx: robot_destroy.wav
                        self.reward_this_step += 1.0
                        self.score += 10
                        self._create_explosion(robot["pos"], 40, self.COLOR_ROBOT)
                        robot["active"] = False
                    
                    # Mark scrap for removal after one hit
                    scrap["pos"].x = -1000 
                    break

    def _update_wave_logic(self):
        game_is_over = self.player_health <= 0 or self.wave_number > self.MAX_WAVES
        if not self.robots and not game_is_over:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
            else:
                if self.wave_number > 0: # Don't reward for clearing wave 0
                    self.reward_this_step += 10.0
                    self.score += 100
                
                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES:
                    return

                self._unlock_content()
                self._start_new_wave()
                self.wave_transition_timer = self.FPS * 3

    def _start_new_wave(self):
        num_robots = 3 + self.wave_number
        base_health = 50 + (self.wave_number * 5)
        base_speed = 0.8 + (self.wave_number // 2) * 0.1
        
        for _ in range(num_robots):
            self.robots.append({
                "pos": pygame.Vector2(self.np_random.integers(50, self.SCREEN_WIDTH - 50), self.np_random.integers(-150, -50)),
                "speed": base_speed + self.np_random.uniform(-0.1, 0.1),
                "health": base_health + self.np_random.uniform(-5, 5),
                "size": self.np_random.integers(25, 41),
                "damage": 10,
                "active": True
            })

    def _unlock_content(self):
        if self.wave_number == 3 and "ice" not in self.unlocked_elements:
            self.unlocked_elements.append("ice")
        if self.wave_number == 5 and "toxic" not in self.unlocked_elements:
            self.unlocked_elements.append("toxic")
        if self.wave_number == 7 and "shatter" not in self.unlocked_combos:
            self.unlocked_combos.append("shatter") # Fire + Ice combo

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)),
                "lifetime": self.np_random.integers(10, 26),
                "color": color,
                "size": self.np_random.integers(2, 6)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "health": self.player_health}

    def _render_game(self):
        # Platform
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, self.player_platform_rect)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 25))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

        # Scraps
        for scrap in self.scraps:
            size = int(scrap["size"])
            color = self.ELEMENT_COLORS[scrap["element"]]
            points = []
            for i in range(5):
                angle = math.radians(scrap["angle"] + i * 72)
                outer_radius = size / 2
                inner_radius = outer_radius / 2
                points.append((int(scrap["pos"].x + outer_radius * math.cos(angle)), int(scrap["pos"].y + outer_radius * math.sin(angle))))
                angle += math.radians(36)
                points.append((int(scrap["pos"].x + inner_radius * math.cos(angle)), int(scrap["pos"].y + inner_radius * math.sin(angle))))

            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Robots
        for robot in self.robots:
            size = int(robot["size"])
            color = self.COLOR_ROBOT_DAMAGED if "damage_timer" in robot else self.COLOR_ROBOT
            rect = pygame.Rect(int(robot["pos"].x - size/2), int(robot["pos"].y - size/2), size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            # Eye
            eye_color = self.ELEMENT_COLORS[robot["last_element_hit"]] if "last_element_hit" in robot else (255,0,0)
            pygame.draw.rect(self.screen, eye_color, (rect.centerx - 5, rect.centery - 2, 10, 4))


        # Player held scrap & charge indicator
        player_center_x = self.player_platform_rect.centerx
        if self.scrap_charge > 0:
            size = 5 + 25 * self.scrap_charge
            element = self.unlocked_elements[self.current_element_idx] if self.unlocked_elements else "none"
            color = self.ELEMENT_COLORS[element]
            pos = (int(player_center_x), int(self.player_platform_rect.top - 20))
            pygame.draw.circle(self.screen, color, pos, int(size/2))
            pygame.draw.circle(self.screen, (255,255,255), pos, int(size/2), 1)

        # Aiming reticle
        ret_x, ret_y = int(self.aim_reticle_pos.x), int(self.aim_reticle_pos.y)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (ret_x - 10, ret_y), (ret_x + 10, ret_y), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (ret_x, ret_y - 10), (ret_x, ret_y + 10), 2)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / 100.0)
        hp_bar_bg = pygame.Rect(10, 10, 200, 20)
        hp_bar_fg = pygame.Rect(10, 10, int(200 * health_pct), 20)
        pygame.draw.rect(self.screen, (50, 0, 0), hp_bar_bg)
        pygame.draw.rect(self.screen, (0, 200, 0), hp_bar_fg)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, hp_bar_bg, 1)

        # Score and Wave
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 35))
        
        wave_text = self.font_large.render(f"WAVE {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 15, 10))
        
        # Element selection
        card_y = self.SCREEN_HEIGHT - 35
        total_width = len(self.unlocked_elements) * 50 - 10
        start_x = self.SCREEN_WIDTH / 2 - total_width / 2
        for i, element in enumerate(self.unlocked_elements):
            card_x = start_x + i * 50
            rect = pygame.Rect(card_x, card_y, 40, 25)
            pygame.draw.rect(self.screen, self.ELEMENT_COLORS[element], rect, border_radius=4)
            if i == self.current_element_idx:
                pygame.draw.rect(self.screen, (255, 255, 0), rect, 3, border_radius=4)
        
        # Wave transition message
        if self.wave_transition_timer > self.FPS:
            msg = f"WAVE {self.wave_number} INCOMING" if self.wave_number > 0 else "GET READY"
            text_surf = self.font_large.render(msg, True, self.COLOR_PLAYER)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
        # Game Over message
        is_victory = self.wave_number > self.MAX_WAVES
        is_terminated = self.player_health <= 0 or is_victory
        if is_terminated:
            msg = "VICTORY!" if is_victory else "GAME OVER"
            text_surf = self.font_large.render(msg, True, (255, 50, 50))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block is for human play and testing, not part of the Gymnasium interface
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    
    pygame.display.set_caption("Scrap Survivor")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    def render_human():
        human_screen.blit(env.screen, (0, 0))
        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    obs, info = env.reset()
    done = False
    
    # To map keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }

    while not done:
        # Human input
        movement_action = 0 # no-op
        space_action = 0
        shift_action = 0
        
        # Check for quit event first
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if done:
            break

        keys = pygame.key.get_pressed()
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break # only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render for human
        render_human()
    
    env.close()