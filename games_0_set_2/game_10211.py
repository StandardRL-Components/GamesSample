import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:06:35.922674
# Source Brief: brief_00211.md
# Brief Index: 211
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for "Beneficial Bacteria".

    The player controls a beneficial bacterium, navigating through a host organism.
    The goal is to reach a target organ while dodging and destroying hostile pathogens.
    The player can move, fire antibiotic projectiles, and flip the direction of gravity.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Fire Antibiotic (0=released, 1=held)
    - actions[2]: Flip Gravity (0=released, 1=held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - an RGB image of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a beneficial bacterium through a hostile environment, destroying pathogens and collecting power-ups to reach the target organ."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire antibiotics and press shift to flip gravity."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (40, 10, 30)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (150, 255, 200)
    COLOR_PATHOGEN_A = (255, 50, 100)
    COLOR_PATHOGEN_B = (200, 50, 255)
    COLOR_ANTIBIOTIC = (100, 200, 255)
    COLOR_POWERUP = (255, 220, 50)
    COLOR_TARGET = (50, 200, 50)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_PROGRESS_BAR = (255, 255, 255)
    COLOR_PROGRESS_BAR_BG = (100, 100, 100)

    # Player settings
    PLAYER_RADIUS = 12
    PLAYER_SPEED = 5.0
    PLAYER_HEALTH_START = 3
    PLAYER_AMMO_START = 20
    PLAYER_SHOOT_COOLDOWN = 6  # frames
    PLAYER_INVINCIBILITY_DURATION = 60 # frames

    # Pathogen settings
    PATHOGEN_RADIUS = 10
    PATHOGEN_SPAWN_PROB_START = 0.03
    PATHOGEN_BASE_SPEED_START = 1.0

    # Antibiotic settings
    ANTIBIOTIC_RADIUS = 4
    ANTIBIOTIC_SPEED = 8.0

    # Powerup settings
    POWERUP_RADIUS = 8
    POWERUP_SPAWN_PROB = 0.005
    POWERUP_AMMO_AMOUNT = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.pathogens = []
        self.antibiotics = []
        self.powerups = []
        self.particles = []
        self.target_area = None
        self.initial_dist_to_target = 1.0
        self.shift_was_held = False
        self.pathogen_spawn_prob = self.PATHOGEN_SPAWN_PROB_START
        self.pathogen_base_speed = self.PATHOGEN_BASE_SPEED_START
        self._bg_elements = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shift_was_held = False
        self.pathogen_spawn_prob = self.PATHOGEN_SPAWN_PROB_START
        self.pathogen_base_speed = self.PATHOGEN_BASE_SPEED_START

        self.player = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT * 0.9),
            "vel": pygame.Vector2(0, 0),
            "radius": self.PLAYER_RADIUS,
            "health": self.PLAYER_HEALTH_START,
            "ammo": self.PLAYER_AMMO_START,
            "shoot_cooldown": 0,
            "invincibility_timer": 0,
            "gravity_dir": 1,  # 1 for down, -1 for up
            "last_move_dir": pygame.Vector2(0, -1)
        }

        self.target_area = pygame.Rect(self.WIDTH / 2 - 50, 10, 100, 20)
        self.initial_dist_to_target = self.player["pos"].distance_to(self.target_area.center)

        self.pathogens = []
        self.antibiotics = []
        self.powerups = []
        self.particles = []

        # Generate a static background for visual interest
        self._bg_elements = []
        for _ in range(50):
            self._bg_elements.append({
                "pos": pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                "radius": self.np_random.integers(50, 151),
                "color": (
                    self.COLOR_BG[0] + self.np_random.integers(5, 16),
                    self.COLOR_BG[1] + self.np_random.integers(5, 16),
                    self.COLOR_BG[2] + self.np_random.integers(5, 16),
                    self.np_random.integers(20, 51)
                )
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. HANDLE INPUT AND UPDATE PLAYER ---
        self._handle_input(action)
        self._update_player()

        # --- 2. UPDATE GAME ENTITIES ---
        self._update_pathogens()
        self._update_projectiles()
        self._update_powerups()
        self._update_particles()
        self._update_difficulty()

        # --- 3. HANDLE COLLISIONS AND CALCULATE REWARD ---
        reward, terminated = self._handle_collisions_and_rewards()
        self.score += reward

        # --- 4. CHECK TERMINATION CONDITIONS ---
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_dir = pygame.Vector2(0, 0)
        if movement == 1: move_dir.y = -1 # Up
        elif movement == 2: move_dir.y = 1 # Down
        elif movement == 3: move_dir.x = -1 # Left
        elif movement == 4: move_dir.x = 1 # Right

        if move_dir.length() > 0:
            self.player["last_move_dir"] = move_dir.copy()
            self.player["vel"] += move_dir * self.PLAYER_SPEED * 0.2
            self._spawn_particles(
                count=1,
                pos=self.player["pos"],
                vel_base=-move_dir * 2,
                vel_rand=1,
                radius_base=3,
                lifespan=10,
                color=self.COLOR_PLAYER_GLOW
            )

        # Shooting
        if space_held and self.player["shoot_cooldown"] == 0 and self.player["ammo"] > 0:
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
            self.player["ammo"] -= 1
            proj_vel = self.player["last_move_dir"].normalize() * self.ANTIBIOTIC_SPEED
            self.antibiotics.append({
                "pos": self.player["pos"].copy(),
                "vel": proj_vel,
                "radius": self.ANTIBIOTIC_RADIUS
            })

        # Gravity Flip
        shift_pressed_this_frame = shift_held and not self.shift_was_held
        if shift_pressed_this_frame:
            self.player["gravity_dir"] *= -1
            self._spawn_particles(
                count=20,
                pos=self.player["pos"],
                vel_base=pygame.Vector2(0,0),
                vel_rand=3,
                radius_base=4,
                lifespan=20,
                color=self.COLOR_POWERUP
            )
        self.shift_was_held = shift_held

    def _update_player(self):
        # Apply gravity
        self.player["vel"].y += 0.2 * self.player["gravity_dir"]
        # Apply drag
        self.player["vel"] *= 0.95
        # Update position
        self.player["pos"] += self.player["vel"]

        # Boundary checks
        self.player["pos"].x = max(self.player["radius"], min(self.WIDTH - self.player["radius"], self.player["pos"].x))
        self.player["pos"].y = max(self.player["radius"], min(self.HEIGHT - self.player["radius"], self.player["pos"].y))

        # Update cooldowns
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        if self.player["invincibility_timer"] > 0:
            self.player["invincibility_timer"] -= 1

    def _update_pathogens(self):
        # Spawn new pathogens
        if self.np_random.random() < self.pathogen_spawn_prob:
            side = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top': pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), -self.PATHOGEN_RADIUS)
            elif side == 'bottom': pos = pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.HEIGHT + self.PATHOGEN_RADIUS)
            elif side == 'left': pos = pygame.Vector2(-self.PATHOGEN_RADIUS, self.np_random.integers(0, self.HEIGHT))
            else: pos = pygame.Vector2(self.WIDTH + self.PATHOGEN_RADIUS, self.np_random.integers(0, self.HEIGHT))
            
            angle_to_center = (pygame.Vector2(self.WIDTH/2, self.HEIGHT/2) - pos).normalize()
            vel = angle_to_center * self.pathogen_base_speed * (0.8 + self.np_random.random() * 0.4)
            
            self.pathogens.append({
                "pos": pos,
                "vel": vel,
                "radius": self.PATHOGEN_RADIUS + self.np_random.integers(-2, 3)
            })

        # Move existing pathogens
        for p in self.pathogens:
            p["pos"] += p["vel"]
        
        # Prune off-screen pathogens
        self.pathogens = [p for p in self.pathogens if -50 < p["pos"].x < self.WIDTH + 50 and -50 < p["pos"].y < self.HEIGHT + 50]

    def _update_projectiles(self):
        for a in self.antibiotics:
            a["pos"] += a["vel"]
        self.antibiotics = [a for a in self.antibiotics if 0 < a["pos"].x < self.WIDTH and 0 < a["pos"].y < self.HEIGHT]

    def _update_powerups(self):
        if self.np_random.random() < self.POWERUP_SPAWN_PROB:
            self.powerups.append({
                "pos": pygame.Vector2(self.np_random.integers(50, self.WIDTH - 49), self.np_random.integers(50, self.HEIGHT - 49)),
                "radius": self.POWERUP_RADIUS,
                "type": "ammo"
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] *= 0.97
        self.particles = [p for p in self.particles if p["lifespan"] > 0 and p["radius"] > 0.5]

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.pathogen_base_speed += 0.05
            self.pathogen_spawn_prob += 0.002

    def _handle_collisions_and_rewards(self):
        reward = 0
        terminated = False

        # Reward for moving closer to target
        current_dist = self.player["pos"].distance_to(self.target_area.center)
        if current_dist < self.player.get("last_dist_to_target", self.initial_dist_to_target):
            reward += 0.01
        self.player["last_dist_to_target"] = current_dist

        # Antibiotic vs Pathogen
        for a in self.antibiotics[:]:
            for p in self.pathogens[:]:
                if a["pos"].distance_to(p["pos"]) < a["radius"] + p["radius"]:
                    self.antibiotics.remove(a)
                    self.pathogens.remove(p)
                    reward += 1.0
                    self._spawn_particles(30, p["pos"], pygame.Vector2(0,0), 2.5, 5, 25, self.COLOR_PATHOGEN_A)
                    break

        # Player vs Pathogen
        if self.player["invincibility_timer"] == 0:
            for p in self.pathogens[:]:
                if self.player["pos"].distance_to(p["pos"]) < self.player["radius"] + p["radius"]:
                    self.pathogens.remove(p)
                    self.player["health"] -= 1
                    self.player["invincibility_timer"] = self.PLAYER_INVINCIBILITY_DURATION
                    reward -= 0.5
                    self._spawn_particles(40, self.player["pos"], pygame.Vector2(0,0), 3, 6, 30, (255, 100, 100))
                    if self.player["health"] <= 0:
                        reward = -100.0
                        terminated = True
                    break
        
        # Player vs Powerup
        for pu in self.powerups[:]:
            if self.player["pos"].distance_to(pu["pos"]) < self.player["radius"] + pu["radius"]:
                self.powerups.remove(pu)
                reward += 5.0
                if pu["type"] == "ammo":
                    self.player["ammo"] += self.POWERUP_AMMO_AMOUNT
                self._spawn_particles(25, pu["pos"], pygame.Vector2(0,0), 2, 4, 20, self.COLOR_POWERUP)
                break
        
        # Player vs Target Area
        if self.target_area.collidepoint(self.player["pos"]):
            reward = 100.0
            terminated = True
            self._spawn_particles(100, self.player["pos"], pygame.Vector2(0,0), 4, 8, 60, self.COLOR_TARGET)

        return reward, terminated

    def _spawn_particles(self, count, pos, vel_base, vel_rand, radius_base, lifespan, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0, vel_rand)
            vel = vel_base + pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": radius_base * self.np_random.uniform(0.8, 1.2),
                "lifespan": lifespan * self.np_random.uniform(0.8, 1.2),
                "color": color
            })

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "ammo": self.player["ammo"],
            "progress": 1.0 - (self.player["pos"].distance_to(self.target_area.center) / self.initial_dist_to_target)
        }
        
    def _render_all(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for bg_elem in self._bg_elements:
            s = pygame.Surface((bg_elem["radius"] * 2, bg_elem["radius"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, bg_elem["color"], (bg_elem["radius"], bg_elem["radius"]), bg_elem["radius"])
            self.screen.blit(s, (bg_elem["pos"].x - bg_elem["radius"], bg_elem["pos"].y - bg_elem["radius"]), special_flags=pygame.BLEND_RGBA_ADD)

        # --- Target Area ---
        pygame.gfxdraw.box(self.screen, self.target_area, (*self.COLOR_TARGET, 50))
        pygame.gfxdraw.rectangle(self.screen, self.target_area, (*self.COLOR_TARGET, 150))

        # --- Particles ---
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"])

        # --- Powerups ---
        for pu in self.powerups:
            pygame.gfxdraw.filled_circle(self.screen, int(pu["pos"].x), int(pu["pos"].y), pu["radius"], self.COLOR_POWERUP)
            pygame.gfxdraw.aacircle(self.screen, int(pu["pos"].x), int(pu["pos"].y), pu["radius"], self.COLOR_POWERUP)

        # --- Pathogens ---
        for p in self.pathogens:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), p["radius"], self.COLOR_PATHOGEN_A)
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), p["radius"], self.COLOR_PATHOGEN_A)

        # --- Antibiotics ---
        for a in self.antibiotics:
            pygame.gfxdraw.filled_circle(self.screen, int(a["pos"].x), int(a["pos"].y), a["radius"], self.COLOR_ANTIBIOTIC)
            pygame.gfxdraw.aacircle(self.screen, int(a["pos"].x), int(a["pos"].y), a["radius"], self.COLOR_ANTIBIOTIC)

        # --- Player ---
        is_invincible = self.player["invincibility_timer"] > 0
        if not (is_invincible and self.steps % 10 < 5):
            # Glow effect
            glow_radius = int(self.player["radius"] * 1.8)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (self.player["pos"].x - glow_radius, self.player["pos"].y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Main body
            pos_int = (int(self.player["pos"].x), int(self.player["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.player["radius"], self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.player["radius"], self.COLOR_PLAYER)

        # --- UI ---
        self._render_ui()

    def _render_ui(self):
        # Health
        health_text = self.font_small.render(f"HEALTH: {self.player['health']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        # Ammo
        ammo_text = self.font_small.render(f"ANTIBIOTICS: {self.player['ammo']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 30))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Progress Bar
        progress = max(0, min(1, 1.0 - (self.player["pos"].distance_to(self.target_area.center) / self.initial_dist_to_target)))
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = self.HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (bar_x, bar_y, bar_width * progress, bar_height), border_radius=4)

        # Game Over Text
        if self.game_over:
            result_text_str = "TARGET REACHED" if self.player["health"] > 0 else "HOST COMPROMISED"
            result_text = self.font_large.render(result_text_str, True, self.COLOR_UI_TEXT)
            text_rect = result_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(result_text, text_rect)

    def close(self):
        pygame.quit()


# Example usage for interactive play
if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It is safe to modify or remove.
    # The render_mode="rgb_array" is used by the environment, but for human play,
    # we will render to a pygame display window.
    
    # Unset the dummy video driver if it was set for headless mode
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window for human play
    pygame.display.set_caption("Beneficial Bacteria")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Map keyboard inputs to action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False # Reset termination flag for new game

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()